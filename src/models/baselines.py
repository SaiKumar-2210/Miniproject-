import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModel:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['paths']['processed_data']

    def load_data(self):
        try:
            df = pd.read_csv(os.path.join(self.processed_path, "features_data.csv"))
            df['date'] = pd.to_datetime(df['date'])
            return df
        except FileNotFoundError:
            logger.error("Feature data not found.")
            return None

    def train_evaluate(self, commodity, district):
        """
        Train ARIMA model for a specific commodity and district.
        """
        df = self.load_data()
        if df is None: return
        
        # Filter data
        subset = df[(df['commodity'] == commodity) & (df['district'] == district)].sort_values('date')
        if subset.empty:
            logger.warning(f"No data for {commodity} in {district}")
            return
        
        # Train/Test Split (Last 30 days as test)
        train = subset.iloc[:-30]
        test = subset.iloc[-30:]
        
        if len(train) < 30:
            logger.warning("Not enough data to train.")
            return

        # Fit ARIMA (Auto-ARIMA logic or fixed order)
        # Using a simple fixed order (5,1,0) for demonstration
        history = [x for x in train['modal_price']]
        predictions = []
        
        logger.info(f"Training ARIMA for {commodity} in {district}...")
        
        # Walk-forward validation
        for t in range(len(test)):
            model = ARIMA(history, order=(5,1,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test.iloc[t]['modal_price']
            history.append(obs)
            
        # Evaluate
        rmse = np.sqrt(mean_squared_error(test['modal_price'], predictions))
        mape = mean_absolute_percentage_error(test['modal_price'], predictions)
        
        logger.info(f"ARIMA Results for {commodity}-{district}: RMSE={rmse:.2f}, MAPE={mape:.2%}")
        return rmse, mape

if __name__ == "__main__":
    config = load_config()
    model = BaselineModel(config)
    
    # Test on one commodity/district
    model.train_evaluate("Rice", "Warangal")
