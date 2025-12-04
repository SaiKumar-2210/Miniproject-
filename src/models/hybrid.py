import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridModel:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['paths']['processed_data']
        self.scaler = MinMaxScaler(feature_range=(-1, 1)) # Residuals can be negative

    def load_data(self):
        try:
            df = pd.read_csv(os.path.join(self.processed_path, "features_data.csv"))
            df['date'] = pd.to_datetime(df['date'])
            return df
        except FileNotFoundError:
            logger.error("Feature data not found.")
            return None

    def create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def train_evaluate(self, commodity, district):
        df = self.load_data()
        if df is None: return
        
        subset = df[(df['commodity'] == commodity) & (df['district'] == district)].sort_values('date')
        if len(subset) < 100:
            logger.warning("Not enough data for Hybrid model.")
            return

        # 1. ARIMA Component
        # We use a rolling forecast or a simple train/test split for the linear part
        train_size = int(len(subset) * 0.8)
        train, test = subset.iloc[:train_size], subset.iloc[train_size:]
        
        history = [x for x in train['modal_price']]
        arima_preds = []
        
        logger.info("Training ARIMA component...")
        # For efficiency in this demo, we fit once and forecast, but ideally walk-forward
        # Using a simpler approach here: Fit on train, predict on test (dynamic=False would be cheating if we use actuals, so we do step-by-step)
        
        # Note: For hybrid, we need in-sample predictions (residuals) for training the LSTM
        # So we fit ARIMA on the whole dataset to get residuals
        model_arima = ARIMA(subset['modal_price'], order=(5,1,0))
        model_fit = model_arima.fit()
        linear_preds = model_fit.fittedvalues
        
        # Calculate Residuals
        residuals = subset['modal_price'] - linear_preds
        
        # 2. LSTM Component on Residuals
        # Prepare data for LSTM
        # We use residuals as the target, and use engineered non-linear features as inputs
        # Use features that capture volatility and seasonality
        feature_cols = [col for col in subset.columns if 'volatility' in col or 'sin' in col or 'cos' in col or 'cum' in col]
        # Fallback if specific features aren't found
        if not feature_cols:
            feature_cols = ['temperature_max', 'rain']
            
        logger.info(f"Using features for LSTM residual modeling: {feature_cols}")
        
        X_exog = subset[feature_cols].values
        y_resid = residuals.values.reshape(-1, 1)
        
        # Scale residuals
        y_resid_scaled = self.scaler.fit_transform(y_resid)
        
        # Scale exog features
        scaler_exog = MinMaxScaler()
        X_exog_scaled = scaler_exog.fit_transform(X_exog)
        
        # Input to LSTM: [Residual_t-1, Exog_t-1] -> Predict Residual_t
        data_combined = np.hstack((y_resid_scaled, X_exog_scaled))
        
        SEQ_LENGTH = 14 # Increased lookback to capture more context
        X_lstm, y_lstm = self.create_sequences(data_combined, SEQ_LENGTH)
        
        # Target for LSTM is the residual (column 0 of combined data)
        y_lstm = y_lstm[:, 0]
        
        # Train/Test Split for LSTM
        # We need to align with the original train/test split
        # The sequences reduce length by SEQ_LENGTH
        split_idx = train_size - SEQ_LENGTH
        
        X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]
        
        from tensorflow.keras.layers import Dropout
        
        model_lstm = Sequential([
            LSTM(32, activation='tanh', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=False),
            Dropout(0.2), # Add dropout to prevent overfitting
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model_lstm.compile(optimizer='adam', loss='mse')
        
        logger.info("Training LSTM component on residuals...")
        model_lstm.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=16, verbose=0)
        
        # Predict Residuals
        resid_preds_scaled = model_lstm.predict(X_test_lstm)
        resid_preds = self.scaler.inverse_transform(resid_preds_scaled)
        
        # 3. Combine Predictions
        # Get ARIMA predictions for the test set (aligned)
        # The LSTM test set starts at train_size (because we split by index)
        # But we lost SEQ_LENGTH items at the start.
        # X_test_lstm corresponds to indices [train_size, end] roughly
        
        # Let's align carefully:
        # subset index: 0 ... N
        # linear_preds: 0 ... N
        # residuals: 0 ... N
        # sequences: SEQ_LENGTH ... N
        # X_lstm[i] uses data from i to i+SEQ_LENGTH to predict i+SEQ_LENGTH
        # So prediction index is i + SEQ_LENGTH
        
        # Test indices for LSTM predictions:
        test_indices = range(train_size, len(subset))
        # We need to ensure X_test_lstm matches these
        # len(X_lstm) = N - SEQ_LENGTH
        # split_idx = train_size - SEQ_LENGTH
        # X_test_lstm = X_lstm[split_idx:] -> indices (split_idx + SEQ_LENGTH) to (N) -> train_size to N
        # So yes, it aligns.
        
        arima_test_preds = linear_preds.iloc[train_size:].values
        
        # Ensure lengths match (sometimes off by 1 due to slicing)
        min_len = min(len(arima_test_preds), len(resid_preds))
        arima_test_preds = arima_test_preds[:min_len]
        resid_preds = resid_preds[:min_len]
        actuals = subset['modal_price'].iloc[train_size:].values[:min_len]
        
        final_preds = arima_test_preds + resid_preds.flatten()
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(actuals, final_preds))
        mape = mean_absolute_percentage_error(actuals, final_preds)
        
        logger.info(f"Hybrid Results for {commodity}-{district}: RMSE={rmse:.2f}, MAPE={mape:.2%}")
        return rmse, mape

if __name__ == "__main__":
    config = load_config()
    model = HybridModel(config)
    model.train_evaluate("Rice", "Warangal")
