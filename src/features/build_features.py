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

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['paths']['processed_data']

    def load_data(self):
        try:
            df = pd.read_csv(os.path.join(self.processed_path, "merged_data.csv"))
            df['date'] = pd.to_datetime(df['date'])
            return df
        except FileNotFoundError:
            logger.error("Merged data not found.")
            return None

    def add_time_features(self, df):
        """
        Add cyclical time features (Sin/Cos).
        """
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df

    def add_lagged_features(self, df):
        """
        Add lagged variables for Price and Weather.
        """
        # Check if we have enough data for lags
        unique_dates = df['date'].nunique()
        if unique_dates <= 1:
            logger.warning(f"Only {unique_dates} unique date(s) in data. Skipping lagged features.")
            return df
        
        # Sort for correct shifting
        df = df.sort_values(by=['commodity', 'district', 'date'])
        
        lags = [1, 7, 14, 30]
        
        for lag in lags:
            # Price Lags
            df[f'price_lag_{lag}'] = df.groupby(['commodity', 'district'])['modal_price'].shift(lag)
            # Arrival Lags - commented out as arrival_quantity column doesn't exist
            # df[f'arrival_lag_{lag}'] = df.groupby(['commodity', 'district'])['arrival_quantity'].shift(lag)
        
        # Weather Lags (Longer term effects)
        weather_lags = [30, 90]
        for lag in weather_lags:
            df[f'temp_max_lag_{lag}'] = df.groupby(['commodity', 'district'])['temperature_max'].shift(lag)
            
        return df

    def add_rolling_features(self, df):
        """
        Add rolling statistics (Volatility, Cumulative Rainfall).
        """
        # Check if we have enough data for rolling features
        unique_dates = df['date'].nunique()
        if unique_dates <= 1:
            logger.warning(f"Only {unique_dates} unique date(s) in data. Skipping rolling features.")
            return df
            
        df = df.sort_values(by=['commodity', 'district', 'date'])
        
        # Volatility (Rolling Std Dev of Price)
        df['price_volatility_7d'] = df.groupby(['commodity', 'district'])['modal_price'].transform(lambda x: x.rolling(window=7).std())
        df['price_volatility_30d'] = df.groupby(['commodity', 'district'])['modal_price'].transform(lambda x: x.rolling(window=30).std())
        
        # Cumulative Rainfall (30 days, 90 days)
        df['rain_cum_30d'] = df.groupby(['commodity', 'district'])['rain'].transform(lambda x: x.rolling(window=30).sum())
        df['rain_cum_90d'] = df.groupby(['commodity', 'district'])['rain'].transform(lambda x: x.rolling(window=90).sum())
        
        # Arrival Momentum (Rolling Mean) - commented out as arrival_quantity column doesn't exist
        # df['arrival_momentum_7d'] = df.groupby(['commodity', 'district'])['arrival_quantity'].transform(lambda x: x.rolling(window=7).mean())
        
        return df

    def add_policy_features(self, df):
        """
        Add interaction terms with MSP.
        """
        # Price to MSP Ratio (How far above/below floor?)
        df['price_to_msp_ratio'] = df['modal_price'] / df['msp_price']
        
        # MSP Floor Flag (Is price near MSP?)
        df['near_msp_floor'] = (df['price_to_msp_ratio'] < 1.1).astype(int)
        
        return df

    def run_pipeline(self):
        df = self.load_data()
        if df is None:
            return
        
        logger.info("Adding time features...")
        df = self.add_time_features(df)
        
        logger.info("Adding lagged features...")
        df = self.add_lagged_features(df)
        
        logger.info("Adding rolling features...")
        df = self.add_rolling_features(df)
        
        logger.info("Adding policy features...")
        df = self.add_policy_features(df)
        
        # Drop rows with NaNs in critical features only (price and price lags)
        # Keep rows even if weather data is missing, but drop if price data is missing
        critical_cols = ['modal_price']
        # Only check price lags if they were created
        price_lag_cols = [col for col in df.columns if 'price_lag' in col]
        if price_lag_cols:
            critical_cols.extend(price_lag_cols)
        
        df_clean = df.dropna(subset=critical_cols)
        
        # Fill remaining NaN values in weather features with 0 or forward fill
        weather_cols = ['temperature_max', 'temperature_min', 'precipitation', 'rain']
        for col in weather_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(0)
        
        # Fill any remaining NaN values in other columns with 0
        df_clean = df_clean.fillna(0)
        
        output_path = os.path.join(self.processed_path, "features_data.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        logger.info(f"Feature engineering complete. Saved to {output_path}")
        logger.info(f"Final shape: {df_clean.shape}")

if __name__ == "__main__":
    config = load_config()
    engineer = FeatureEngineer(config)
    engineer.run_pipeline()
