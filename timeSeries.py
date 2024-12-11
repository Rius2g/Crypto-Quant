import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
import time 
from prophet import Prophet
import requests


class CryptoDataFetcher:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
    
    def get_price_history(self, coin_id, days=365):
        """
        Fetches the price history for the given coin in the given day range
        """
        endpoint = f"{self.base_url}/coins/{coin_id}/market_chart"

        params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"
                }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data["prices"], columns=["timestamp", "close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            return df

        except requests.exceptions.RequestException as e:
            print(e)
            return None


    def get_current_price(self, coin_id):
        """
        Fetches the current price of the given coin
        """

        endpoint = f"{self.base_url}/simple/price"

        params = {
                "ids": coin_id,
                "vs_currencies": "usd"
                }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            return data[coin_id]["usd"]

        except requests.exceptions.RequestException as e:
            print(e)
            return None


class CryptoBuySignal:
    def __init__(self):
        self.model = Prophet(
                changepoint_prior_scale=0.15,
                seasonality_prior_scale=10,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                )
        self.scaler = None
        self.price_scale = 1.0

    def prepare_data(self, df):
        prophet_df = df.copy()
        prophet_df["ds"] = pd.to_datetime(prophet_df["timestamp"])

        self.price_scale = prophet_df["close"].mean()
        if self.price_scale == 0:
            self.price_scale = 1.0

        prophet_df["y"] = prophet_df["close"] / self.price_scale
        print(f"Debug - Mean price: ${self.price_scale:,.2f}")
        return prophet_df[["ds", "y"]]


    def train(self, df):
        data = self.prepare_data(df)
        self.model.fit(data)

    def calculate_buy_signal(self, current_price, df):
        """
        Calculate buy signal with proper price scaling
        """
        days_back = 60
        days_forward = 14

        date_list = pd.date_range(
                start=datetime.now() - timedelta(days=days_back),
                end=datetime.now() + timedelta(days=days_forward),
                freq="D"
                )

        future = pd.DataFrame({"ds": date_list})
        forecast = self.model.predict(future)
        
        # Scale predictions back to original price range
        for col in ['yhat', 'yhat_lower', 'yhat_upper']:
            forecast[col] = forecast[col] * self.price_scale
            print(f"Debug - {col} first value: ${forecast[col].iloc[0]:,.2f}")

        current_date = datetime.now().date()
        current_forecast = forecast[forecast['ds'].dt.date == current_date].iloc[0]

        # Debugging price ranges
        print(f"Debug - Current price: ${current_price:,.2f}")
        print(f"Debug - Predicted range: ${current_forecast['yhat_lower']:,.2f} to ${current_forecast['yhat_upper']:,.2f}")

        # Price position score (0-40)
        predicted_price = current_forecast["yhat"]
        lower_bound = current_forecast["yhat_lower"]
        upper_bound = current_forecast["yhat_upper"]
        
        # Calculate price position
        price_range = upper_bound - lower_bound
        if price_range > 0:
            price_position = (current_price - lower_bound) / price_range
            price_score = max(0, min(40, (1 - price_position) * 40))
        else:
            price_score = 20

        # Trend score (0-30)
        future_data = forecast[forecast['ds'].dt.date > current_date]
        first_price = future_data['yhat'].iloc[0]
        last_price = future_data['yhat'].iloc[-1]
        trend_percent = ((last_price - first_price) / first_price) * 100
        
        # Adjusted trend scoring
        trend_score = max(0, min(30, trend_percent * 2))

        # Rest of the scoring logic remains the same...
        recent_data = df.tail(days_back)
        recent_predictions = forecast[forecast['ds'].dt.date <= current_date].tail(days_back)
        mape = np.mean(np.abs((recent_data["close"].values - recent_predictions["yhat"].values) / recent_data["close"].values))
        accuracy_score = max(0, min(15, (1-mape) * 15))

        current_ma = df["close"].tail(7).mean()
        previous_ma = df["close"].tail(14).head(7).mean()
        momentum = ((current_ma - previous_ma) / previous_ma) * 100
        momentum_score = max(0, min(15, momentum if momentum > 0 else 0))

        total_score = price_score + trend_score + accuracy_score + momentum_score

        return {
                "buy_signal": round(total_score),
                "price_score": round(price_score, 2),
                "trend_score": round(trend_score, 2),
                "accuracy_score": round(accuracy_score, 2),
                "momentum_score": round(momentum_score, 2),
                "forecast": {
                    "current_predicted": round(predicted_price, 2),
                    "current_lower": round(lower_bound, 2),
                    "current_upper": round(upper_bound, 2),
                    "trend": round(trend_percent, 2)
                    }
                } 
                
class CryptoProphetAnalyzer:
    def __init__(self):
        self.data_fetcher = CryptoDataFetcher()

    def analyze_crypto(self, coin_id):
        historical_data = self.data_fetcher.get_price_history(coin_id)
        if historical_data is None:
            return None

        time.sleep(1)

        current_price = self.data_fetcher.get_current_price(coin_id)
        if current_price is None:
            return None

        self.buy_signal_model = CryptoBuySignal()
        self.buy_signal_model.train(historical_data)

        signal = self.buy_signal_model.calculate_buy_signal(current_price, historical_data)

        return {
                "coin_id": coin_id,
                "current_price": current_price,
                "analysis": signal
                }

def main():
    analyzer = CryptoProphetAnalyzer()

    coins = ["bitcoin", "ethereum", "dogecoin", "cardano", "ripple", "chainlink", "solana"]

    for coin in coins:
        print(f"\nAnalyzing {coin.title()}...")

        time.sleep(60)
        try:
            result = analyzer.analyze_crypto(coin)


            if result:
                print(f"Current Price: ${result['current_price']:,.2f}")
                print(f"Buy Signal: {result['analysis']['buy_signal']}/100")
                print(f"\nComponent Scores:")
                print(f"- Price Position: {result['analysis']['price_score']}/40")
                print(f"- Trend: {result['analysis']['trend_score']}/30")
                print(f"- Accuracy: {result['analysis']['accuracy_score']}/15")
                print(f"- Momentum: {result['analysis']['momentum_score']}/15")
                
                forecast = result['analysis']['forecast']
                print(f"\nForecast:")
                print(f"- Predicted Price: ${forecast['current_predicted']:,.2f}")
                print(f"- Price Range: ${forecast['current_lower']:,.2f} - ${forecast['current_upper']:,.2f}")
                print(f"- Trend: {'Upward' if forecast['trend'] > 0 else 'Downward'}")

        except Exception as e:
            print(e)
            print("Error analyzing coin. Skipping...")
            if "429" in str(e):
                print("API limit reached. Exiting...")
                time.sleep(60)
        
if __name__ == "__main__":
    main()
