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

    def predict(self, commodity, district, forecast_days=3):
        """
        Generate prediction for the next N days (default 3).
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
        
        # 4. Multi-day Inference
        from datetime import timedelta
        last_date = history_df['date'].max()
        forecasts_list = []
        
        if model_type == 'lstm':
            feature_cols = ['modal_price', 'arrival_quantity', 'temperature_max', 'rain', 'month_sin', 'month_cos']
            SEQ_LENGTH = 14
            latest_data = features_df[feature_cols].tail(SEQ_LENGTH).values.copy()
            
            if len(latest_data) < SEQ_LENGTH:
                return {"error": "Not enough data after feature engineering"}
            
            scaled_data = scaler.transform(latest_data)
            
            for day_offset in range(1, forecast_days + 1):
                X_input = scaled_data.reshape(1, SEQ_LENGTH, len(feature_cols))
                pred_scaled = model.predict(X_input, verbose=0)
                
                dummy = np.zeros((1, len(feature_cols)))
                dummy[:, 0] = pred_scaled.flatten()
                pred_price = scaler.inverse_transform(dummy)[0, 0]
                
                forecast_date = last_date + timedelta(days=day_offset)
                forecasts_list.append({
                    'date': forecast_date.strftime('%Y-%m-%d'),
                    'price': float(pred_price)
                })
                
                # Shift window: drop first row, append new predicted row
                new_row_scaled = scaled_data[-1].copy()
                new_row_scaled[0] = pred_scaled.flatten()[0]
                scaled_data = np.vstack([scaled_data[1:], new_row_scaled.reshape(1, -1)])
            
        elif model_type == 'arima':
            forecast_values = model.forecast(steps=forecast_days)
            for day_offset, pred_price in enumerate(forecast_values, start=1):
                forecast_date = last_date + timedelta(days=day_offset)
                forecasts_list.append({
                    'date': forecast_date.strftime('%Y-%m-%d'),
                    'price': float(pred_price)
                })
        else:
            return {"error": "Model type not supported"}
        
        result['modal_price'] = forecasts_list[0]['price']  # T+1 as the primary
        result['forecasts'] = forecasts_list
        
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
            
        # 6. Extract recent history
        if history_df is not None and not history_df.empty:
            recent_history = history_df.tail(10)[['date', 'modal_price']].copy()
            recent_history['date'] = recent_history['date'].dt.strftime('%Y-%m-%d')
            result['history'] = recent_history.to_dict(orient='records')
        else:
            result['history'] = []
        
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
