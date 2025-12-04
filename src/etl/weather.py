import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import logging
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WeatherClient:
    def __init__(self, config):
        self.config = config
        self.cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        self.retry_session = retry(self.cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = self.retry_session)
        self.url = "https://archive-api.open-meteo.com/v1/archive"

    def fetch_historical_weather(self, lat, lon, start_date, end_date):
        """
        Fetch historical weather data for a specific location.
        """
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "rain_sum"],
            "timezone": "Asia/Kolkata"
        }
        
        try:
            responses = self.openmeteo.weather_api(self.url, params=params)
            response = responses[0]
            
            # Process daily data
            daily = response.Daily()
            daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
            daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()
            daily_precipitation_sum = daily.Variables(2).ValuesAsNumpy()
            daily_rain_sum = daily.Variables(3).ValuesAsNumpy()
            
            daily_data = {"date": pd.date_range(
                start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
                end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = daily.Interval()),
                inclusive = "left"
            )}
            
            daily_data["temperature_max"] = daily_temperature_2m_max
            daily_data["temperature_min"] = daily_temperature_2m_min
            daily_data["precipitation"] = daily_precipitation_sum
            daily_data["rain"] = daily_rain_sum
            
            df = pd.DataFrame(data = daily_data)
            df['latitude'] = lat
            df['longitude'] = lon
            return df
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return pd.DataFrame()

    def fetch_for_districts(self, start_date, end_date):
        """
        Fetch weather for all configured districts.
        Note: In a real app, we'd need a mapping of District -> Lat/Lon.
        Using a placeholder mapping for now.
        """
        # Placeholder coordinates for Telangana districts
        district_coords = {
            "Warangal": (17.9689, 79.5941),
            "Karimnagar": (18.4386, 79.1288),
            "Nalgonda": (17.0577, 79.2684),
            "Khammam": (17.2473, 80.1514),
            "Nizamabad": (18.6725, 78.0941)
        }
        
        all_data = []
        for district in self.config['region']['districts']:
            if district in district_coords:
                lat, lon = district_coords[district]
                logger.info(f"Fetching weather for {district}...")
                df = self.fetch_historical_weather(lat, lon, start_date, end_date)
                df['district'] = district
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def save_data(self, df, filename="weather_data.csv"):
        path = os.path.join(self.config['paths']['raw_data'], filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Weather data saved to {path}")

if __name__ == "__main__":
    config = load_config()
    client = WeatherClient(config)
    
    # Fetch last 30 days
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    df = client.fetch_for_districts(start_date, end_date)
    
    if not df.empty:
        print(df.head())
        client.save_data(df)
    else:
        print("No weather data fetched.")
