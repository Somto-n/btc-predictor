"""
Walk-forward backtesting engine.
Primary output: max consecutive losing streaks per model, per direction (UP/DOWN).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BacktestResult:
    name:               str
    all_preds:          list = field(default_factory=list)   # predicted labels
    all_actuals:        list = field(default_factory=list)   # true labels
    all_probs:          list = field(default_factory=list)   # confidence scores

    # filled by analyse()
    accuracy:           float = 0.0
    up_accuracy:        float = 0.0     # accuracy when model predicted UP
    down_accuracy:      float = 0.0     # accuracy when model predicted DOWN

    max_consec_loss_all:  int = 0
    max_consec_loss_up:   int = 0       # consecutive losses on UP predictions only
    max_consec_loss_down: int = 0       # consecutive losses on DOWN predictions only
    avg_consec_loss:    float = 0.0

    streak_pcts: dict = field(default_factory=dict)   # p50/p75/p90/p95/p99


def _max_streak(flags: list[int]) -> int:
    """Return the longest run of 1s in a binary list."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _streak_percentiles(flags: list[int]) -> dict:
    """Collect all individual losing streak lengths and return percentiles."""
    streaks, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        else:
            if cur > 0:
                streaks.append(cur)
            cur = 0
    if cur > 0:
        streaks.append(cur)
    if not streaks:
        return {p: 0 for p in [50, 75, 90, 95, 99]}
    a = np.array(streaks)
    return {p: int(np.percentile(a, p)) for p in [50, 75, 90, 95, 99]}


def analyse(result: BacktestResult) -> BacktestResult:
    preds   = np.array(result.all_preds)
    actuals = np.array(result.all_actuals)
    correct = (preds == actuals).astype(int)
    wrong   = 1 - correct

    result.accuracy = correct.mean()

    # direction-specific masks
    up_mask   = preds == 1
    down_mask = preds == 0

    if up_mask.sum() > 0:
        result.up_accuracy   = correct[up_mask].mean()
        result.max_consec_loss_up   = _max_streak(wrong[up_mask].tolist())
    if down_mask.sum() > 0:
        result.down_accuracy = correct[down_mask].mean()
        result.max_consec_loss_down = _max_streak(wrong[down_mask].tolist())

    result.max_consec_loss_all = _max_streak(wrong.tolist())
    result.streak_pcts         = _streak_percentiles(wrong.tolist())

    # average losing streak length
    streaks, cur = [], 0
    for w in wrong.tolist():
        if w:
            cur += 1
        else:
            if cur:
                streaks.append(cur)
            cur = 0
    if cur:
        streaks.append(cur)
    result.avg_consec_loss = float(np.mean(streaks)) if streaks else 0.0

    return result


def walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    train_window: int,
    test_window: int,
    step: int,
    build_model_fn: Callable,
    model_name: str,
    sequence_len: int = 0,   # >0 for LSTM
) -> BacktestResult:
    """
    Rolling walk-forward validation.
    Trains on [i : i+train_window], tests on [i+train_window : i+train_window+test_window].
    Slides by `step` each iteration.
    """
    result = BacktestResult(name=model_name)
    X      = df[feature_cols].values
    y      = df["target"].values
    n      = len(df)
    fold   = 0

    for start in range(0, n - train_window - test_window, step):
        tr_end = start + train_window
        te_end = tr_end + test_window

        X_train, y_train = X[start:tr_end], y[start:tr_end]
        X_test,  y_test  = X[tr_end:te_end], y[tr_end:te_end]

        model = build_model_fn(sequence_len=sequence_len)

        if sequence_len > 0:
            # LSTM: reshape to (samples, timesteps, features)
            Xtr3 = _make_sequences(X_train, sequence_len)
            ytr3 = y_train[sequence_len:]
            Xte3 = _make_sequences(X_test,  sequence_len)
            yte3 = y_test[sequence_len:]
            if len(Xtr3) == 0 or len(Xte3) == 0:
                continue
            model.fit(Xtr3, ytr3)
            probs = model.predict(Xte3).flatten()
            preds = (probs > 0.5).astype(int)
            result.all_probs.extend(probs.tolist())
            result.all_preds.extend(preds.tolist())
            result.all_actuals.extend(yte3.tolist())
        else:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = model.predict(X_test)
            result.all_probs.extend(probs.tolist())
            result.all_preds.extend(preds.tolist())
            result.all_actuals.extend(y_test.tolist())

        fold += 1

    print(f"  {model_name}: {fold} folds, {len(result.all_preds)} predictions")
    return analyse(result)


def _make_sequences(X: np.ndarray, seq_len: int):
    """Convert 2D array into 3D sequences for LSTM input."""
    if len(X) <= seq_len:
        return np.empty((0, seq_len, X.shape[1]))
    return np.array([X[i:i+seq_len] for i in range(len(X) - seq_len)])
