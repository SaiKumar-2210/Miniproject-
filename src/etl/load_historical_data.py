import logging
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config
from src.etl.agmarknet import AgmarknetClient
from src.etl.weather import WeatherClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_historical_data():
    config = load_config()
    
    # Initialize clients
    agmarknet = AgmarknetClient(config)
    weather = WeatherClient(config)
    
    # Define date range (2 years)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
    
    logger.info(f"Fetching data from {start_date} to {end_date}...")
    
    # 1. Fetch Price Data
    logger.info("Fetching price data...")
    # Note: AgmarknetClient.fetch_data generates mock data if no API key
    # We need to ensure it generates for the full range, so we force mock data
    prices_df = agmarknet.fetch_data(date_from=start_date, date_to=end_date, use_mock=True)
    
    if not prices_df.empty:
        agmarknet.save_data(prices_df)
        logger.info(f"Price data fetched: {len(prices_df)} rows")
    else:
        logger.error("Failed to fetch price data")
        
    # 2. Fetch Weather Data
    logger.info("Fetching weather data...")
    weather_df = weather.fetch_for_districts(start_date, end_date)
    
    if not weather_df.empty:
        weather.save_data(weather_df)
        logger.info(f"Weather data fetched: {len(weather_df)} rows")
    else:
        logger.error("Failed to fetch weather data")
        
    logger.info("Historical data loading complete.")

if __name__ == "__main__":
    load_historical_data()
