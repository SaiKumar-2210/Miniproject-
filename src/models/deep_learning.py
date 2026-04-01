import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import logging
import os
import sys
import pickle

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.utils.config_loader import load_config
from src.utils.model_registry import ModelRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Data Quality Constants ────────────────────────────────────────────
MIN_REQUIRED_ROWS = 365          # At least 1 full seasonal cycle
MAX_NAN_RATIO     = 0.10         # No feature may be >10% empty
REQUIRED_FEATURES = ['modal_price']  # Must always exist
# Weather/lag features are strongly recommended but not hard-required,
# because they depend on a successful ETL run.
RECOMMENDED_FEATURES = ['temperature_max', 'rain', 'price_lag_7']
SEQ_LENGTH = 14                  # Look-back window (2 weeks)
TEST_SIZE  = 30                  # Last 30 days for evaluation


class DeepLearningModel:
    def __init__(self, config):
        self.config = config
        self.processed_path = config['paths']['processed_data']
        self.models_path = os.path.join(config['paths']['models'], 'lstm')
        os.makedirs(self.models_path, exist_ok=True)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.registry = ModelRegistry()

    # ── Load ──────────────────────────────────────────────────────────
    def load_data(self):
        try:
            df = pd.read_csv(os.path.join(self.processed_path, "features_data.csv"))
            df['date'] = pd.to_datetime(df['date'])
            return df
        except FileNotFoundError:
            logger.error("Feature data not found.")
            return None

    # ── Sequence Builder ──────────────────────────────────────────────
    def create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            xs.append(data[i:(i + seq_length)])
            ys.append(data[i + seq_length])
        return np.array(xs), np.array(ys)

    # ── Data Quality Gate ─────────────────────────────────────────────
    def _validate_data(self, subset, commodity, district):
        """
        Strict data validation. Returns (clean_subset, feature_cols) on
        success, or (None, None) if data is unfit for training.
        """
        tag = f"{commodity}-{district}"

        # 1. Minimum row count
        if len(subset) < MIN_REQUIRED_ROWS:
            logger.error(
                f"ABORT [{tag}]: Only {len(subset)} rows. "
                f"Need at least {MIN_REQUIRED_ROWS}."
            )
            return None, None

        # 2. Build dynamic feature list
        exclude_cols = ['date', 'commodity', 'district', 'state', 'market', 'year',
                        'source', 'arrival_date']
        numeric_cols = subset.select_dtypes(include=[np.number]).columns.tolist()

        # Drop columns that are 100 % empty
        all_nan_cols = [c for c in numeric_cols if subset[c].isna().all()]
        if all_nan_cols:
            logger.warning(f"[{tag}] Dropping 100%% empty columns: {all_nan_cols}")
            numeric_cols = [c for c in numeric_cols if c not in all_nan_cols]

        feature_cols = ['modal_price'] + [
            c for c in numeric_cols
            if c not in exclude_cols and c != 'modal_price'
        ]

        # 3. Check required features exist
        for feat in REQUIRED_FEATURES:
            if feat not in feature_cols:
                logger.error(f"ABORT [{tag}]: Required feature '{feat}' is missing.")
                return None, None

        # 4. Warn about missing recommended features
        for feat in RECOMMENDED_FEATURES:
            if feat not in feature_cols:
                logger.warning(
                    f"[{tag}] Recommended feature '{feat}' missing. "
                    "Model accuracy may be reduced."
                )

        # 5. Forward-fill small gaps then drop remaining NaN rows
        subset = subset.copy()
        subset[feature_cols] = subset[feature_cols].ffill()
        subset = subset.dropna(subset=feature_cols)

        # 6. Re-check row count after NaN removal
        if len(subset) < MIN_REQUIRED_ROWS:
            logger.error(
                f"ABORT [{tag}]: Only {len(subset)} clean rows remain "
                f"after NaN cleanup. Need {MIN_REQUIRED_ROWS}."
            )
            return None, None

        # 7. Check per-column NaN ratio (should be 0 after cleanup, but safety net)
        for col in feature_cols:
            ratio = subset[col].isna().sum() / len(subset)
            if ratio > MAX_NAN_RATIO:
                logger.error(
                    f"ABORT [{tag}]: Column '{col}' is {ratio*100:.1f}%% empty "
                    f"(max allowed: {MAX_NAN_RATIO*100:.0f}%%). Fix ETL pipeline."
                )
                return None, None

        logger.info(
            f"[{tag}] Data quality OK: {len(subset)} rows, "
            f"{len(feature_cols)} features."
        )
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

        data = subset[feature_cols].values

        # Scale
        scaled_data = self.scaler.fit_transform(data)

        # Sequences
        X, y = self.create_sequences(scaled_data, SEQ_LENGTH)
        y = y[:, 0]  # target = modal_price (column 0)

        # Train / Test split (last 30 days)
        if len(X) <= TEST_SIZE:
            logger.warning("Not enough sequenced data for 30-day test set.")
            return

        train_size = len(X) - TEST_SIZE
        X_train_full, X_test = X[:train_size], X[train_size:]
        y_train_full, y_test = y[:train_size], y[train_size:]

        # Train / Validation split (90 / 10)
        val_idx = int(len(X_train_full) * 0.9)
        X_train, X_val = X_train_full[:val_idx], X_train_full[val_idx:]
        y_train, y_val = y_train_full[:val_idx], y_train_full[val_idx:]

        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss', patience=15,
            restore_best_weights=True, verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-5, verbose=1
        )

        # Strategy (TPU / GPU / CPU)
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
            model = Sequential([
                Input(shape=(X_train.shape[1], X_train.shape[2])),
                Bidirectional(GRU(32, return_sequences=False)),
                BatchNormalization(),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='huber'
            )

        logger.info(f"Training GRU for {commodity} in {district}...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=32, epochs=100, verbose=0,
            callbacks=[early_stop, reduce_lr]
        )
        logger.info(f"Stopped after {len(history.history['loss'])} epochs.")

        # Predict & inverse-transform
        preds_scaled = model.predict(X_test)
        dummy = np.zeros((len(preds_scaled), len(feature_cols)))
        dummy[:, 0] = preds_scaled.flatten()
        inverse_preds = self.scaler.inverse_transform(dummy)[:, 0]

        dummy_y = np.zeros((len(y_test), len(feature_cols)))
        dummy_y[:, 0] = y_test
        inverse_y = self.scaler.inverse_transform(dummy_y)[:, 0]

        # Evaluate
        rmse = np.sqrt(mean_squared_error(inverse_y, inverse_preds))
        mape = mean_absolute_percentage_error(inverse_y, inverse_preds)
        logger.info(f"GRU Results for {commodity}-{district}: RMSE={rmse:.2f}, MAPE={mape:.2%}")

        # Save model + scaler
        model_filename = f"{commodity}_{district}_lstm.keras"
        model_path = os.path.join(self.models_path, model_filename)
        model.save(model_path)

        scaler_filename = f"{commodity}_{district}_scaler.pkl"
        scaler_path = os.path.join(self.models_path, scaler_filename)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        logger.info(f"Saved model to {model_path}")

        # Register
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        self.registry.register_model(
            commodity=commodity,
            district=district,
            model_type="lstm",
            model_path=os.path.relpath(model_path, project_root),
            scaler_path=os.path.relpath(scaler_path, project_root),
            metrics={"rmse": rmse, "mape": mape}
        )
        return rmse, mape

    # ── Batch Runner ──────────────────────────────────────────────────
    def run_training(self):
        commodities = self.config['commodities']
        districts = self.config['region']['districts']
        for commodity in commodities:
            for district in districts:
                logger.info(f"Starting training pipeline for {commodity} - {district}")
                try:
                    self.train_evaluate(commodity, district)
                except Exception as e:
                    logger.error(f"Training failed for {commodity}-{district}: {e}")


if __name__ == "__main__":
    config = load_config()
    model = DeepLearningModel(config)
    model.run_training()
