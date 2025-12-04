import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
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

class DeepLearningModel:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['paths']['processed_data']
        self.scaler = MinMaxScaler(feature_range=(0, 1))

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
            logger.warning("Not enough data for DL model.")
            return

        # Features to use
        feature_cols = ['modal_price', 'arrival_quantity', 'temperature_max', 'rain', 'month_sin', 'month_cos']
        data = subset[feature_cols].values
        
        # Scale data
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        SEQ_LENGTH = 14
        X, y = self.create_sequences(scaled_data, SEQ_LENGTH)
        
        # Target is price (column 0)
        y = y[:, 0]
        
        # Train/Test Split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build LSTM Model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        logger.info(f"Training LSTM for {commodity} in {district}...")
        model.fit(X_train, y_train, batch_size=16, epochs=20, verbose=0) # verbose=0 for cleaner logs
        
        # Predict
        predictions = model.predict(X_test)
        
        # Inverse transform predictions
        # We need to create a dummy array to inverse transform because scaler expects all features
        dummy_array = np.zeros((len(predictions), len(feature_cols)))
        dummy_array[:, 0] = predictions.flatten()
        inverse_preds = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        dummy_y = np.zeros((len(y_test), len(feature_cols)))
        dummy_y[:, 0] = y_test
        inverse_y = self.scaler.inverse_transform(dummy_y)[:, 0]
        
        # Evaluate
        rmse = np.sqrt(mean_squared_error(inverse_y, inverse_preds))
        mape = mean_absolute_percentage_error(inverse_y, inverse_preds)
        
        logger.info(f"LSTM Results for {commodity}-{district}: RMSE={rmse:.2f}, MAPE={mape:.2%}")
        return rmse, mape

if __name__ == "__main__":
    config = load_config()
    model = DeepLearningModel(config)
    model.train_evaluate("Rice", "Warangal")
