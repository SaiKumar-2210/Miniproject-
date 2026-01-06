import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging
import os
import sys

import pickle
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config
from src.utils.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModel:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['paths']['processed_data']
        self.models_path = os.path.join(config['paths']['models'], 'arima')
        os.makedirs(self.models_path, exist_ok=True)
        self.registry = ModelRegistry()

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
        
        # Refit on entire dataset for production (optional, but recommended for future dates)
        # For now, we will just save the last fitted model from the walk-forward or fit a new one on all data
        # Fitting on all data for deployment:
        logger.info(f"Retraining ARIMA on full dataset for deployment...")
        final_model = ARIMA(subset['modal_price'].values, order=(5,1,0))
        final_model_fit = final_model.fit()
        
        # Save Model
        model_filename = f"{commodity}_{district}_arima.pkl"
        model_path = os.path.join(self.models_path, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(final_model_fit, f)
            
        logger.info(f"Saved ARIMA model to {model_path}")
        
        # Register Model
        self.registry.register_model(
            commodity=commodity,
            district=district,
            model_type="arima",
            model_path=os.path.relpath(model_path, os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) # Relative to project root roughly or just use abs path? Registry logic handles abs/rel. 
            # Let's use relative path from project root for portability
        )
        # Re-calc relative path cleanly
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        rel_path = os.path.relpath(model_path, project_root)
        
        self.registry.register_model(
            commodity=commodity,
            district=district,
            model_type="arima",
            model_path=rel_path,
            metrics={"rmse": rmse, "mape": mape}
        )

        return rmse, mape

if __name__ == "__main__":
    config = load_config()
    model = BaselineModel(config)
    
    # Test on one commodity/district
    model.train_evaluate("Rice", "Warangal")
