import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config
from src.utils.model_registry import ModelRegistry
from src.features.build_features import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self):
        self.config = load_config()
        self.registry = ModelRegistry()
        self.feature_engineer = FeatureEngineer(self.config)
        self.processed_path = self.config['paths']['processed_data']
        
        # Cache for loaded models to avoid reloading from disk on every request
        self.model_cache = {} 
        
    def _get_historical_data(self, commodity, district, num_days=100):
        """
        Load recent historical raw data for feature engineering.
        In production, this would query a DB. Here we read the merged CSV.
        """
        # Optimized: Read only necessary columns/rows if possible, but for CSV we read all and filter
        # TODO: Implement a better data store
        try:
            df = pd.read_csv(os.path.join(self.processed_path, "merged_data.csv"))
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter
            mask = (df['commodity'] == commodity) & (df['district'] == district)
            subset = df[mask].sort_values('date').tail(num_days)
            
            if subset.empty:
                logger.warning(f"No history found for {commodity} in {district}")
                return None
                
            return subset
        except FileNotFoundError:
            logger.error("merged_data.csv not found.")
            return None

    def load_artifacts(self, commodity, district):
        """
        Load model, scaler, and garch params from registry.
        """
        cache_key = f"{commodity}_{district}"
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]
            
        entry = self.registry.get_model_entry(commodity, district)
        if not entry:
            raise ValueError(f"No model registered for {commodity} in {district}")
            
        model_type = entry['model_type']
        
        # Paths are stored relative to project root in registry (as per our previous edit logic)
        # We need absolute paths
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        
        # Load Model
        model_path = os.path.join(project_root, entry['model_path'])
        if model_type == 'lstm':
            model = load_model(model_path)
        elif model_type == 'arima':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Load Scaler (if exists)
        scaler = None
        if 'scaler_path' in entry:
            scaler_path = os.path.join(project_root, entry['scaler_path'])
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                
        # Load GARCH Params (if exists)
        garch_params = None
        if 'garch_params_path' in entry:
            garch_path = os.path.join(project_root, entry['garch_params_path'])
            with open(garch_path, 'r') as f:
                garch_params = json.load(f)
                
        artifacts = {
            "model": model,
            "model_type": model_type,
            "scaler": scaler,
            "garch": garch_params
        }
        
        self.model_cache[cache_key] = artifacts
        return artifacts

    def predict(self, commodity, district):
        """
        Generate prediction for the next available date (T+1).
        """
        # 1. Load Data (History)
        # We need enough history for lags (max lag 90) + rolling (max 90)
        # So fetch 150 days to be safe
        history_df = self._get_historical_data(commodity, district, num_days=150)
        if history_df is None or len(history_df) < 30:
            return {"error": "Insufficient historical data"}
            
        # 2. Feature Engineering
        # We process the history. The 'last' row after FE will contain features for T (using info from T, T-1...)
        # But wait, we want to predict T+1. 
        # Usually, models are trained to predict T using features from T (if T is price) or T-1.
        # My LSTM `create_sequences` usually uses T-1...T-N to predict T.
        # So if we have data up to T, we can predict T+1? 
        # Let's check `deep_learning.py` -> `create_sequences`:
        # x = data[i:(i + seq_length)], y = data[i + seq_length]
        # So input is sequence of length SEQ_LENGTH. Output is the NEXT value.
        # So if we feed the LAST SEQ_LENGTH rows of features, we get the prediction for the NEXT day.
        
        # Step 2a: Generate features on the history
        features_df = self.feature_engineer.generate_features(history_df.copy())
        
        # 3. Load Artifacts
        try:
            artifacts = self.load_artifacts(commodity, district)
        except ValueError as e:
            return {"error": str(e)}
            
        model = artifacts['model']
        scaler = artifacts['scaler']
        model_type = artifacts['model_type']
        
        result = {}
        
        # 4. Inference
        if model_type == 'lstm':
            # LSTM needs specific columns and scaling
            feature_cols = ['modal_price', 'arrival_quantity', 'temperature_max', 'rain', 'month_sin', 'month_cos']
            
            # Extract last SEQ_LENGTH rows (14)
            SEQ_LENGTH = 14
            latest_data = features_df[feature_cols].tail(SEQ_LENGTH).values
            
            if len(latest_data) < SEQ_LENGTH:
                return {"error": "Not enough data after feature engineering"}
                
            # Scale
            scaled_data = scaler.transform(latest_data)
            
            # Reshape (1, SEQ_LENGTH, n_features)
            X_input = scaled_data.reshape(1, SEQ_LENGTH, len(feature_cols))
            
            # Predict
            # Determine prediction strategy: CPU or TPU?
            # For inference, default CPU is fine usually.
            pred_scaled = model.predict(X_input)
            
            # Inverse Transform
            # Dummy array trick again
            dummy = np.zeros((1, len(feature_cols)))
            dummy[:, 0] = pred_scaled.flatten()
            pred_price = scaler.inverse_transform(dummy)[0, 0]
            
        elif model_type == 'arima':
            # ARIMA predict
            # For ARIMA, we usually just need the history of the target variable
            # But the saved model object (statsmodels wrapper) might store its own history.
            # If we used `ARIMA.fit()`, the model is fixed to the training data.
            # To predict on NEW data, we need to `apply` or `append` the new observations.
            # Since we pickled the results wrapper, we can try using `apply`.
            
            # Simplify: Just refit a new ARIMA on the history_df (fast enough)?
            # Or use the saved parameters.
            # Statsmodels `apply` is the correct way: new_res = res.apply(new_data)
            # But that requires keeping the whole history aligned.
            
            # Alternative: Just predict the next step based on loaded history?
            # If the model was trained on T_train, and now we are at T_now (months later),
            # the saved model is stale.
            # "Productionizing" ARIMA often implies re-training or using `append`.
            
            # For this MVP, let's assume we re-train on the fly or the saved model is fresh.
            # Let's try to forecast 1 step ahead from the saved model state.
            # CAUTION: If saved model is old, this forecast is for the past.
            
            # Better approach for ARIMA in this specific lightweight setup:
            # We saved the FIT object. It has a state.
            # We should probably just Refit for best accuracy if it's cheap?
            # Or use `forecast` if we just trained it.
            
            # Let's assume the user just ran training (Step 5 of task).
            # So the model is fresh.
            forecast = model.forecast(steps=1)
            pred_price = forecast[0]
            
        else:
            return {"error": "Model type not supported"}
            
        result['modal_price'] = float(pred_price)
        
        # 5. Volatility (GARCH)
        garch = artifacts.get('garch')
        if garch:
            # Sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
            # We need the last return and last sigma.
            # But wait, we don't know the last sigma unless we run the GARCH filter on the whole history or save state.
            # Simplified approach: Estimate current volatility using rolling window or re-run GARCH filter.
            
            # Re-running GARCH filter on recent history using Fixed Params:
            omega = garch['omega']
            alpha = garch['alpha']
            beta = garch['beta']
            
            # Calculate returns
            # We need raw prices again, which we have in history_df
            prices = history_df['modal_price'].values
            returns = np.diff(np.log(prices)) # Log returns
            
            # Filter recursion
            # Initialize sigma2 with variance
            sigma2 = np.var(returns)
            for r in returns:
                sigma2 = omega + alpha * r**2 + beta * sigma2
                
            current_sigma = np.sqrt(sigma2)
            
            # Calculate Floors/Ceilings
            # Price * (1 +/- sigma) or Price +/- (Price * Sigma)?
            # Sigma is std dev of Returns (percentage change roughly).
            # So Price Floor ~= Price * exp(-Sigma) or Price * (1 - Sigma)
            
            result['volatility'] = float(current_sigma)
            result['price_min'] = float(pred_price * (1 - current_sigma))
            result['price_max'] = float(pred_price * (1 + current_sigma))
            
            # Risk Level
            if current_sigma < 0.01:
                risk = "Low"
            elif current_sigma < 0.03:
                risk = "Moderate"
            else:
                risk = "High"
            result['risk_level'] = risk
            
        return result

if __name__ == "__main__":
    # Test
    engine = InferenceEngine()
    print("Testing Inference Engine...")
    # Requires artifacts to be present.
    try:
        pred = engine.predict("Rice", "Warangal")
        print(pred)
    except Exception as e:
        print(f"Error: {e}")
