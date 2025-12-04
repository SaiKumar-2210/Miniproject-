import pandas as pd
import numpy as np
import logging
import os
import sys
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Diagnostics:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['paths']['processed_data']
        self.output_path = os.path.join(config['paths']['models'], 'diagnostics')
        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self):
        try:
            df = pd.read_csv(os.path.join(self.processed_path, "features_data.csv"))
            df['date'] = pd.to_datetime(df['date'])
            return df
        except FileNotFoundError:
            logger.error("Features data not found.")
            return None

    def check_stationarity(self, df, commodity, district):
        """
        Perform Augmented Dickey-Fuller test.
        """
        subset = df[(df['commodity'] == commodity) & (df['district'] == district)].sort_values('date')
        if len(subset) < 30:
            logger.warning(f"Not enough data for ADF test for {commodity}-{district}")
            return
            
        series = subset['modal_price']
        result = adfuller(series.dropna())
        
        logger.info(f"\n--- Stationarity Check (ADF Test) for {commodity} in {district} ---")
        logger.info(f"ADF Statistic: {result[0]}")
        logger.info(f"p-value: {result[1]}")
        logger.info("Critical Values:")
        for key, value in result[4].items():
            logger.info(f"\t{key}: {value}")
            
        if result[1] > 0.05:
            logger.warning("Series is likely NON-STATIONARY (p-value > 0.05). Differencing needed.")
        else:
            logger.info("Series is likely STATIONARY.")

    def analyze_residuals(self, df, commodity, district):
        """
        Train a simple ARIMA model and analyze residuals.
        """
        subset = df[(df['commodity'] == commodity) & (df['district'] == district)].sort_values('date')
        if len(subset) < 50:
            return

        # Split 80/20
        train_size = int(len(subset) * 0.8)
        train = subset['modal_price'][:train_size]
        test = subset['modal_price'][train_size:]
        
        # Train ARIMA (1,1,1) as a baseline
        try:
            model = ARIMA(train, order=(1,1,1))
            model_fit = model.fit()
            
            residuals = model_fit.resid
            
            logger.info(f"\n--- Residual Analysis for {commodity} in {district} ---")
            logger.info(f"Residual Mean: {residuals.mean()}")
            logger.info(f"Residual Std: {residuals.std()}")
            
            # Check for white noise (Ljung-Box test could be added, but simple stats first)
            
            # Plot residuals
            plt.figure(figsize=(10, 6))
            plt.plot(residuals)
            plt.title(f'ARIMA Residuals for {commodity}-{district}')
            plt.savefig(os.path.join(self.output_path, f'residuals_{commodity}_{district}.png'))
            plt.close()
            logger.info(f"Residual plot saved to {self.output_path}")
            
        except Exception as e:
            logger.error(f"Error in residual analysis: {e}")

    def verify_split(self, df, commodity, district):
        """
        Verify chronological split.
        """
        subset = df[(df['commodity'] == commodity) & (df['district'] == district)].sort_values('date')
        if len(subset) < 10:
            return
            
        train_size = int(len(subset) * 0.8)
        train = subset.iloc[:train_size]
        test = subset.iloc[train_size:]
        
        logger.info(f"\n--- Split Verification for {commodity} in {district} ---")
        logger.info(f"Train Date Range: {train['date'].min().date()} to {train['date'].max().date()}")
        logger.info(f"Test Date Range: {test['date'].min().date()} to {test['date'].max().date()}")
        
        if train['date'].max() < test['date'].min():
            logger.info("Split is strictly CHRONOLOGICAL (Correct).")
        else:
            logger.error("Split is NOT chronological (Potential Data Leakage!).")

    def run_diagnostics(self):
        df = self.load_data()
        if df is None:
            return
            
        # Run for a few representative pairs
        pairs = [
            ('Maize', 'Warangal'),
            ('Cotton', 'Warangal'),
            ('Rice', 'Khammam') # If available
        ]
        
        for commodity, district in pairs:
            # Check if pair exists
            if not df[(df['commodity'] == commodity) & (df['district'] == district)].empty:
                self.check_stationarity(df, commodity, district)
                self.verify_split(df, commodity, district)
                self.analyze_residuals(df, commodity, district)
            else:
                logger.warning(f"No data for {commodity} in {district}")

if __name__ == "__main__":
    config = load_config()
    diagnostics = Diagnostics(config)
    diagnostics.run_diagnostics()
