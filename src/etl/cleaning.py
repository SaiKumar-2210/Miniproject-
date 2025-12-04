import pandas as pd
import numpy as np
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, config):
        self.config = config
        self.raw_path = config['paths']['raw_data']
        self.processed_path = config['paths']['processed_data']

    def load_raw_data(self):
        try:
            prices = pd.read_csv(os.path.join(self.raw_path, "agmarknet_data.csv"))
            weather = pd.read_csv(os.path.join(self.raw_path, "weather_data.csv"))
            msp = pd.read_csv(os.path.join(self.raw_path, "msp_data.csv"))
            return prices, weather, msp
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return None, None, None

    def clean_prices(self, df):
        """Clean price data."""
        # Handle column name variations
        if 'arrival_date' in df.columns:
            df = df.rename(columns={'arrival_date': 'date'})
        
        logger.info(f"Prices shape before cleaning: {df.shape}")
        df['date'] = pd.to_datetime(df['date'])
        # Remove timezone information if present
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        
        # Sort by date
        df = df.sort_values(by=['commodity', 'district', 'date']).reset_index(drop=True)
        
        # Forward fill missing prices within each commodity-district group
        df['modal_price'] = df.groupby(['commodity', 'district'])['modal_price'].transform(lambda x: x.ffill())
        
        logger.info(f"Prices shape after cleaning: {df.shape}")
        return df

    def clean_weather(self, df):
        """Clean weather data."""
        df['date'] = pd.to_datetime(df['date'])
        # Remove timezone information if present
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        df = df.sort_values(by=['district', 'date']).reset_index(drop=True)
        
        # Interpolate missing weather data
        numeric_cols = ['temperature_max', 'temperature_min', 'precipitation', 'rain']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df.groupby('district')[col].transform(lambda x: x.interpolate(method='linear'))
        
        return df

    def merge_data(self, prices, weather, msp):
        """Merge all datasets."""
        # Merge Prices and Weather on Date and District
        merged = pd.merge(prices, weather, on=['date', 'district'], how='left')
        
        # Extract year from date
        merged['year'] = merged['date'].dt.year
        
        # Melt MSP to be long format: Year, Commodity, MSP_Value
        msp_long = msp.melt(id_vars=['year'], var_name='commodity', value_name='msp_price')
        
        final_df = pd.merge(merged, msp_long, on=['year', 'commodity'], how='left')
        
        return final_df

    def run_pipeline(self):
        prices, weather, msp = self.load_raw_data()
        if prices is None:
            return
        
        logger.info("Cleaning prices...")
        prices_clean = self.clean_prices(prices)
        
        logger.info("Cleaning weather...")
        weather_clean = self.clean_weather(weather)
        
        logger.info("Merging datasets...")
        final_df = self.merge_data(prices_clean, weather_clean, msp)
        
        output_path = os.path.join(self.processed_path, "merged_data.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_df.to_csv(output_path, index=False)
        logger.info(f"Merged data saved to {output_path}")
        logger.info(f"Final shape: {final_df.shape}")

if __name__ == "__main__":
    config = load_config()
    cleaner = DataCleaner(config)
    cleaner.run_pipeline()
