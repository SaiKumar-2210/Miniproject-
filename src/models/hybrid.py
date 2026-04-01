import pandas as pd
import numpy as np
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Data Quality Constants (mirror deep_learning.py) ──────────────────
MIN_REQUIRED_ROWS = 365
MAX_NAN_RATIO     = 0.10
SEQ_LENGTH = 14
TEST_SIZE  = 30


class HybridModel:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['paths']['processed_data']
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # Residuals can be negative

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
            xs.append(data[i:(i + seq_length)])
            ys.append(data[i + seq_length])
        return np.array(xs), np.array(ys)

    # ── Data Quality Gate ─────────────────────────────────────────────
    def _validate_data(self, subset, commodity, district):
        """
        Returns (clean_subset, feature_cols) or (None, None) if data
        is unfit for training.
        """
        tag = f"{commodity}-{district}"

        if len(subset) < MIN_REQUIRED_ROWS:
            logger.error(
                f"ABORT [{tag}]: Only {len(subset)} rows. "
                f"Need at least {MIN_REQUIRED_ROWS}."
            )
            return None, None

        # Build feature list (for residual modelling — exclude modal_price itself)
        exclude_cols = ['date', 'commodity', 'district', 'state', 'market',
                        'year', 'source', 'arrival_date', 'modal_price']
        numeric_cols = subset.select_dtypes(include=[np.number]).columns.tolist()

        # Drop 100% empty columns
        all_nan = [c for c in numeric_cols if subset[c].isna().all()]
        if all_nan:
            logger.warning(f"[{tag}] Dropping 100%% empty columns: {all_nan}")
            numeric_cols = [c for c in numeric_cols if c not in all_nan]

        feature_cols = [c for c in numeric_cols if c not in exclude_cols]

        if not feature_cols:
            logger.error(f"ABORT [{tag}]: No usable features for residual LSTM.")
            return None, None

        # Forward-fill small gaps, then drop remaining NaN rows
        subset = subset.copy()
        subset[feature_cols] = subset[feature_cols].ffill()
        subset = subset.dropna(subset=feature_cols + ['modal_price'])

        if len(subset) < MIN_REQUIRED_ROWS:
            logger.error(
                f"ABORT [{tag}]: Only {len(subset)} clean rows after NaN cleanup."
            )
            return None, None

        logger.info(f"[{tag}] Data quality OK: {len(subset)} rows, {len(feature_cols)} exog features.")
        return subset, feature_cols

    # ── Train & Evaluate ──────────────────────────────────────────────
    def train_evaluate(self, commodity, district):
        df = self.load_data()
        if df is None:
            return

        subset = df[
            (df['commodity'] == commodity) & (df['district'] == district)
        ].sort_values('date')

        # ── DATA QUALITY GATE ──
        subset, feature_cols = self._validate_data(subset, commodity, district)
        if subset is None:
            return

        if len(subset) <= TEST_SIZE + SEQ_LENGTH:
            logger.warning("Not enough data for Hybrid model 30-day test.")
            return

        train_size = len(subset) - TEST_SIZE

        # ─────────────────────── 1. ARIMA Component ───────────────────
        logger.info("Training ARIMA component...")
        model_arima = ARIMA(subset['modal_price'], order=(5, 1, 0))
        model_fit = model_arima.fit()
        linear_preds = model_fit.fittedvalues

        # Residuals = Actual − ARIMA prediction
        residuals = subset['modal_price'] - linear_preds

        # ─────────────────────── 2. GRU on Residuals ──────────────────
        logger.info(f"Using {len(feature_cols)} features for GRU residual modeling.")

        X_exog = subset[feature_cols].values
        y_resid = residuals.values.reshape(-1, 1)

        # Scale
        y_resid_scaled = self.scaler.fit_transform(y_resid)
        scaler_exog = MinMaxScaler()
        X_exog_scaled = scaler_exog.fit_transform(X_exog)

        data_combined = np.hstack((y_resid_scaled, X_exog_scaled))

        X_lstm, y_lstm = self.create_sequences(data_combined, SEQ_LENGTH)
        y_lstm = y_lstm[:, 0]  # target = residual (col 0)

        # Align with original train/test boundary
        split_idx = len(X_lstm) - TEST_SIZE
        X_train_full, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train_full, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

        # 90/10 train/val
        val_idx = int(len(X_train_full) * 0.9)
        X_train, X_val = X_train_full[:val_idx], X_train_full[val_idx:]
        y_train, y_val = y_train_full[:val_idx], y_train_full[val_idx:]

        early_stop = EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-5, verbose=1
        )

        # Strategy
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            logger.info(f"Running on TPU: {tpu.master()}")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
        except ValueError:
            logger.info("TPU not found, using default strategy (CPU/GPU)")
            strategy = tf.distribute.get_strategy()

        # ── GRU Model (lighter, better for ≤4 yr datasets) ──
        with strategy.scope():
            model_gru = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                Bidirectional(GRU(32, return_sequences=False)),
                BatchNormalization(),
                Dropout(0.1),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model_gru.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse'
            )

        logger.info("Training GRU component on residuals...")
        history = model_gru.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100, batch_size=32, verbose=0,
            callbacks=[early_stop, reduce_lr]
        )

        # Predict residuals
        resid_preds_scaled = model_gru.predict(X_test)
        resid_preds = self.scaler.inverse_transform(resid_preds_scaled)

        # ─────────────────────── 3. Combine ───────────────────────────
        arima_test_preds = linear_preds.iloc[-TEST_SIZE:].values

        min_len = min(len(arima_test_preds), len(resid_preds))
        arima_test_preds = arima_test_preds[:min_len]
        resid_preds = resid_preds[:min_len]
        actuals = subset['modal_price'].iloc[train_size:].values[:min_len]

        final_preds = arima_test_preds + resid_preds.flatten()

        # Evaluate
        rmse = np.sqrt(mean_squared_error(actuals, final_preds))
        mape = mean_absolute_percentage_error(actuals, final_preds)
        accuracy = max(0.0, 100.0 - (mape * 100.0))

        logger.info(f"Hybrid Results for {commodity}-{district}: RMSE={rmse:.2f}, Accuracy={accuracy:.1f}%")
        
        # Plot Performance
        plt.figure(figsize=(10, 6))
        plt.plot(actuals, label='Actual Price', color='blue', linewidth=2)
        plt.plot(final_preds, label='Hybrid Predicted Price', color='orange', linestyle='--', linewidth=2)
        plt.title(f'Hybrid (ARIMA+GRU) Predictions vs Actuals ({commodity} - {district})')
        plt.xlabel('Days (Test Set)')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        models_path = os.path.join(self.config['paths']['models'], 'hybrid')
        os.makedirs(models_path, exist_ok=True)
        plot_path = os.path.join(models_path, f"{commodity}_{district}_hybrid_plot.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved performance plot to {plot_path}")
        
        return rmse, mape


if __name__ == "__main__":
    config = load_config()
    model = HybridModel(config)
    model.train_evaluate("Rice", "Warangal")
