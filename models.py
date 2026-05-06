"""
Model factory functions. Each returns a fresh, unfitted model instance.
All sklearn-compatible (have fit / predict / predict_proba).
LSTM wrapped in a sklearn-compatible class.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from catboost import CatBoostClassifier


# ─── Tree / linear models ─────────────────────────────────────────────────────

def make_logistic(**kwargs) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=500, random_state=42)),
    ])


def make_random_forest(**kwargs) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )


def make_xgboost(**kwargs) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )


def make_catboost(**kwargs) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=0,
    )


# ─── LSTM (sklearn-compatible wrapper) ────────────────────────────────────────

class LSTMClassifier:
    """Thin wrapper around a Keras LSTM so it fits the walk_forward interface."""

    def __init__(self, sequence_len: int = 60, n_features: int = 1,
                 epochs: int = 10, batch_size: int = 64):
        self.sequence_len = sequence_len
        self.n_features   = n_features
        self.epochs       = epochs
        self.batch_size   = batch_size
        self._model       = None

    def _build(self):
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        tf.get_logger().setLevel("ERROR")

        m = Sequential([
            LSTM(64, input_shape=(self.sequence_len, self.n_features), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1,  activation="sigmoid"),
        ])
        m.compile(optimizer="adam", loss="binary_crossentropy")
        return m

    def fit(self, X: np.ndarray, y: np.ndarray):
        # X shape: (samples, sequence_len, n_features)
        self.n_features = X.shape[2]
        self._model = self._build()
        self._model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X, verbose=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - p, p])


def make_lstm(sequence_len: int = 60, **kwargs) -> LSTMClassifier:
    return LSTMClassifier(sequence_len=sequence_len, epochs=10, batch_size=64)


# ─── Registry used by compare.py ─────────────────────────────────────────────

MODEL_REGISTRY = [
    {"name": "Logistic Regression", "fn": make_logistic,     "is_lstm": False},
    {"name": "Random Forest",       "fn": make_random_forest, "is_lstm": False},
    {"name": "XGBoost",             "fn": make_xgboost,       "is_lstm": False},
    {"name": "CatBoost",            "fn": make_catboost,      "is_lstm": False},
    {"name": "LSTM",                "fn": make_lstm,           "is_lstm": True},
]
