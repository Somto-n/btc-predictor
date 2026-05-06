# BTC/USDT 5-Minute Direction Predictor

An end-to-end machine learning pipeline that predicts the direction of the next BTC/USDT 5-minute candle using an XGBoost + CatBoost ensemble, with a live signal terminal for real-time trade signals.

## Results (6-Month Out-of-Sample Backtest)

| Metric | Value |
|---|---|
| Test period | Oct 2024 – Apr 2025 |
| Candles tested | 52,000+ |
| UP signal accuracy | **65.2%** |
| Signal rate (All Combined filter) | ~3.9% of candles |
| Cap-out rate (Martingale, 6-step) | < 1.5% |
| Back-to-back cap-outs | Never occurred in 6 months |

## Pipeline Overview

```
Coinbase REST API
      │
      ▼
 data_feed.py       ← Fetch 5-min OHLCV (60 days)
      │
      ▼
 features.py        ← Engineer 15+ technical indicators
      │
      ▼
 models.py          ← Train XGBoost + CatBoost
      │
      ▼
 extended_backtest.py ← 6-month walk-forward backtest + Martingale simulation
      │
      ▼
 live_signal.py     ← Real-time signal terminal (polls every 5 min)
```

## Features Engineered

From raw OHLCV data:

- **Momentum:** RSI(14), MACD(12/26/9), Williams %R, Stochastic %K/%D, ROC, CCI
- **Trend:** EMA crossovers (9/21), ADX(14), Supertrend
- **Volatility:** Bollinger Bands (%B + width), ATR(14)
- **Volume:** OBV (normalised), volume delta (vs rolling average)
- **Structure:** Candle body ratio, 15-min EMA trend, 1-hour EMA trend

All features computed with walk-forward cross-validation to eliminate lookahead bias.

## Signal Filter (All Combined)

A signal is only fired when all conditions are met:

```python
# Both models must agree on direction
signal = xgb_pred == cat_pred

# Trend must be strong
adx > 25

# Active trading hours only (UTC)
7 <= hour_utc < 21

# Multi-timeframe EMA confluence
15-min EMA and 1-hour EMA must align with signal direction
```

## Live Signal Terminal

```bash
conda activate btc_pred
python live_signal.py
```

Trains models at startup, then polls Coinbase every 5 minutes. Outputs:

```
══════════════════════════════════════════════════════
  [14:55 UTC]  SIGNAL FIRED
  Direction    : UP  ▲
  Confidence   : XGB 68%  |  CAT 71%
  Stake        : $5.00  (Step 1 of 6)
  Bankroll     : $500.00
  Session P&L  : +$12.50
══════════════════════════════════════════════════════
  After this candle closes, enter result:  [W]in / [L]oss / [S]kip:
```

Tracks Martingale state in real time — doubles stake on loss, resets on win or cap-out.

## Installation

```bash
conda create -n btc_pred python=3.10
conda activate btc_pred
pip install pandas numpy pandas-ta xgboost catboost scikit-learn requests matplotlib
```

## Project Structure

| File | Purpose |
|---|---|
| `config.py` | Shared settings (thresholds, symbol, timeframe) |
| `data_feed.py` | Fetch 5-min OHLCV from Coinbase REST API |
| `features.py` | Feature engineering (15+ indicators) |
| `models.py` | XGBoost + CatBoost model definitions |
| `backtest.py` | Walk-forward backtest engine |
| `compare.py` | Model comparison + charts |
| `extended_backtest.py` | 6-month backtest with Martingale simulation |
| `loss_recovery.py` | Recovery length analysis |
| `live_signal.py` | Live signal terminal |
