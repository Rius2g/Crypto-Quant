import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
import time 
from prophet import Prophet
import requests


class CryptoDataFetcher:
    def __init__(self):
        self.base_url = "https://api.coingecko.com/api/v3"
    
    def get_volume_history(self, coin_id, days=365):
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

            df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            return df

        except requests.exceptions.RequestException as e:
            print(e)
            return None

    def get_current_volume(self, coin_id):
        """
        Fetches the current 24h trading volume
        """
        endpoint = f"{self.base_url}/coins/{coin_id}/market_chart"
        
        params = {
            "vs_currency": "usd",
            "days": "1",
            "interval": "daily"
        }

        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Get the most recent volume
            current_volume = data["total_volumes"][0][1]
            return current_volume

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
        self.volume_scale= 1.0

    def prepare_data(self, df):
        prophet_df = df.copy()
        prophet_df["ds"] = pd.to_datetime(prophet_df["timestamp"])

        self.volue_scale = prophet_df["volume"].median()
        if self.volume_scale == 0:
            self.volume_scale = 1.0

        prophet_df["y"] = prophet_df["volume"] / self.volume_scale

        print(f"Debug - Median volume: ${self.volume_scale:,.2f}, First scaled value: {prophet_df['y'].iloc[0]:,.4f}")

        return prophet_df[["ds", "y"]]


    def train(self, df):
        data = self.prepare_data(df)
        self.model.fit(data)

    def calculate_volume_signal(self, current_volume, df):
        """
        Calculate buy signal based on volume patterns with enhanced error checking
        """
        try:
            days_back = 30 
            days_forward = 7

            date_list = pd.date_range(
                    start=datetime.now() - timedelta(days=days_back),
                    end=datetime.now() + timedelta(days=days_forward),
                    freq="D"
                    )

            future = pd.DataFrame({"ds": date_list})
            forecast = self.model.predict(future)
            
            print(f"Debug - Scaling factor: {self.volume_scale:,.2f}")
            
            # Scale predictions back to original volume range
            for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                forecast[col] = forecast[col] * self.volume_scale
                print(f"Debug - {col} first value: ${forecast[col].iloc[0]:,.2f}")

            current_date = datetime.now().date()
            current_forecast = forecast[forecast['ds'].dt.date == current_date].iloc[0]

            # Volume position score (0-40)
            predicted_volume = current_forecast["yhat"]
            normal_range = current_forecast["yhat_upper"] - current_forecast["yhat_lower"]
            
            print(f"Debug - Predicted volume: ${predicted_volume:,.2f}")
            print(f"Debug - Current volume: ${current_volume:,.2f}")
            
            volume_position = (current_volume - predicted_volume) / normal_range
            volume_score = max(0, min(40, (volume_position + 1) * 20))

            # Volume trend score (0-30)
            future_data = forecast[forecast['ds'].dt.date > current_date]
            first_volume = future_data['yhat'].iloc[0]
            last_volume = future_data['yhat'].iloc[-1]
            trend_percent = ((last_volume - first_volume) / first_volume) * 100
            trend_score = max(0, min(30, trend_percent))

            # Accuracy score (0-15)
            recent_data = df.tail(days_back)
            recent_predictions = forecast[forecast['ds'].dt.date <= current_date].tail(days_back)
            mape = np.mean(np.abs((recent_data["volume"].values - recent_predictions["yhat"].values) / recent_data["volume"].values))
            accuracy_score = max(0, min(15, (1-mape) * 15))

            # Volume momentum (0-15)
            current_ma = df["volume"].tail(7).mean()
            previous_ma = df["volume"].tail(14).head(7).mean()
            momentum = ((current_ma - previous_ma) / previous_ma) * 100
            momentum_score = max(0, min(15, momentum if momentum > 0 else 0))

            # Calculate total score
            total_score = volume_score + trend_score + accuracy_score + momentum_score

            print(f"Debug - Scores:")
            print(f"Volume Position Score: {volume_score}")
            print(f"Trend Score: {trend_score}")
            print(f"Accuracy Score: {accuracy_score}")
            print(f"Momentum Score: {momentum_score}")

            return {
                    "buy_signal": round(total_score),
                    "volume_score": round(volume_score, 2),
                    "trend_score": round(trend_score, 2),
                    "accuracy_score": round(accuracy_score, 2),
                    "momentum_score": round(momentum_score, 2),
                    "forecast": {
                        "current_predicted": round(predicted_volume, 2),
                        "current_lower": round(current_forecast["yhat_lower"], 2),
                        "current_upper": round(current_forecast["yhat_upper"], 2),
                        "trend": round(trend_percent, 2)
                        },
                    "debug": {
                        "volume_position": volume_position,
                        "normal_range": normal_range,
                        "scale_factor": self.volume_scale,
                        "trend_percent": trend_percent,
                        "mape": mape,
                        "momentum": momentum,
                        "current_ma": current_ma,
                        "previous_ma": previous_ma
                    }
            }
            
        except Exception as e:
            print(f"Error in calculate_volume_signal: {str(e)}")
            raise

   
class CryptoVolumeAnalyzer:
    def __init__(self):
        self.data_fetcher = CryptoDataFetcher()

    def analyze_crypto(self, coin_id):
        try:
            historical_data = self.data_fetcher.get_volume_history(coin_id)
            if historical_data is None:
                print(f"No historical data available for {coin_id}")
                return None

            time.sleep(1)

            current_volume = self.data_fetcher.get_current_volume(coin_id)
            if current_volume is None:
                print(f"No current volume data available for {coin_id}")
                return None

            print(f"Debug - Current volume: ${current_volume:,.2f}")
            print(f"Debug - Historical data points: {len(historical_data)}")

            volume_model = CryptoBuySignal()
            volume_model.train(historical_data)

            signal = volume_model.calculate_volume_signal(current_volume, historical_data)

            return {
                "coin_id": coin_id,
                "current_volume": current_volume,
                "analysis": signal
            }

        except Exception as e:
            print(f"Error in analyze_crypto for {coin_id}: {str(e)}")
            return None

               
def main():
    analyzer = CryptoVolumeAnalyzer()
    coins = ["bitcoin", "ethereum", "dogecoin", "cardano", "ripple", "chainlink", "solana"]

    for coin in coins:
        print(f"\n{'='*50}")
        print(f"Analyzing {coin.title()} Volume...")
        time.sleep(60)

        try:
            result = analyzer.analyze_crypto(coin)
            
            if result and result['analysis']:  # Add check for analysis
                print(f"\nDebug info:")
                print(f"Coin ID: {result['coin_id']}")
                print(f"Current Volume: ${result['current_volume']:,.2f}")
                
                print(f"\nVolume Signal: {result['analysis']['buy_signal']}/100")
                print(f"\nComponent Scores:")
                print(f"- Volume Position: {result['analysis']['volume_score']}/40")
                print(f"- Volume Trend: {result['analysis']['trend_score']}/30")
                print(f"- Accuracy: {result['analysis']['accuracy_score']}/15")
                print(f"- Volume Momentum: {result['analysis']['momentum_score']}/15")
                
                forecast = result['analysis']['forecast']
                print(f"\nVolume Forecast:")
                print(f"- Predicted Volume: ${forecast['current_predicted']:,.2f}")
                print(f"- Normal Range: ${forecast['current_lower']:,.2f} - ${forecast['current_upper']:,.2f}")
                print(f"- Trend: {'Increasing' if forecast['trend'] > 0 else 'Decreasing'}")

            time.sleep(3)

        except Exception as e:
            print(f"Error analyzing {coin} (with exception details): {str(e)}")
            if "429" in str(e):
                print("Rate limit hit, waiting 60 seconds...")
                time.sleep(60)

if __name__ == "__main__":
    main()
