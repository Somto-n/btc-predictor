"""
Technical indicator feature engineering on 5-min BTC OHLCV data.
Target: 1 if next candle closes higher, 0 if lower.
Uses the 'ta' library (pip install ta) — compatible with pandas 3.x.
"""

import pandas as pd
import numpy as np
import ta

FEATURE_COLS: list = []   # filled after build_features()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  raw OHLCV DataFrame (columns: time, open, high, low, close, volume)
    Output: DataFrame with all indicator columns + binary target (no lookahead)
    """
    d = df.copy()
    o = d["open"].astype(float)
    h = d["high"].astype(float)
    l = d["low"].astype(float)
    c = d["close"].astype(float)
    v = d["volume"].astype(float)

    # ── Momentum ───────────────────────────────────────────────────────────────
    d["rsi_14"]   = ta.momentum.RSIIndicator(c, window=14).rsi()
    d["rsi_30"]   = ta.momentum.RSIIndicator(c, window=30).rsi()
    d["roc_12"]   = ta.momentum.ROCIndicator(c, window=12).roc()
    d["willr_14"] = ta.momentum.WilliamsRIndicator(h, l, c, lbp=14).williams_r()
    stoch         = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    d["stoch_k"]  = stoch.stoch()
    d["stoch_d"]  = stoch.stoch_signal()

    # ── MACD ───────────────────────────────────────────────────────────────────
    macd          = ta.trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
    d["macd"]     = macd.macd()
    d["macd_sig"] = macd.macd_signal()
    d["macd_hist"]= macd.macd_diff()

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    bb            = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    d["bb_upper"] = bb.bollinger_hband()
    d["bb_lower"] = bb.bollinger_lband()
    d["bb_pct"]   = bb.bollinger_pband()    # %B: position within bands
    d["bb_width"] = bb.bollinger_wband()    # bandwidth

    # ── Volatility ─────────────────────────────────────────────────────────────
    d["atr_14"]   = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    kc            = ta.volatility.KeltnerChannel(h, l, c, window=20)
    d["kc_pct"]   = kc.keltner_channel_pband()

    # ── Trend ─────────────────────────────────────────────────────────────────
    adx           = ta.trend.ADXIndicator(h, l, c, window=14)
    d["adx_14"]   = adx.adx()
    d["dmp_14"]   = adx.adx_pos()
    d["dmn_14"]   = adx.adx_neg()
    d["cci_20"]   = ta.trend.CCIIndicator(h, l, c, window=20).cci()
    d["ema_9"]    = ta.trend.EMAIndicator(c, window=9).ema_indicator()
    d["ema_21"]   = ta.trend.EMAIndicator(c, window=21).ema_indicator()
    d["ema_50"]   = ta.trend.EMAIndicator(c, window=50).ema_indicator()
    d["ema_cross_9_21"]  = (d["ema_9"]  > d["ema_21"]).astype(int)
    d["ema_cross_21_50"] = (d["ema_21"] > d["ema_50"]).astype(int)

    # ── Volume ─────────────────────────────────────────────────────────────────
    d["obv"]          = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    vol_ma            = v.rolling(20).mean()
    d["volume_ratio"] = v / (vol_ma + 1e-9)

    # ── Aroon (14) ────────────────────────────────────────────────────────────
    aroon              = ta.trend.AroonIndicator(h, l, window=14)
    d["aroon_up"]      = aroon.aroon_up()
    d["aroon_down"]    = aroon.aroon_down()
    d["aroon_osc"]     = aroon.aroon_indicator()   # aroon_up - aroon_down

    # Breakout flag: aroon_up/down hits 100 after being ≤80 previous candle
    # (shift(1) is safe — purely past data, no lookahead)
    d["aroon_up_break"]   = ((d["aroon_up"]   == 100) &
                             (d["aroon_up"].shift(1)   <= 80)).astype(int)
    d["aroon_down_break"] = ((d["aroon_down"] == 100) &
                             (d["aroon_down"].shift(1) <= 80)).astype(int)

    # ── Candle structure ───────────────────────────────────────────────────────
    body               = (c - o).abs()
    d["candle_body_ratio"] = body / (d["atr_14"] + 1e-9)
    d["candle_direction"]  = (c > o).astype(int)

    # ── Target (shift -1 — next candle direction, NO lookahead) ───────────────
    d["target"] = (c.shift(-1) > c).astype(int)

    # ── Drop NaN rows + last row (no target) ──────────────────────────────────
    d.dropna(inplace=True)
    d = d.iloc[:-1].reset_index(drop=True)

    # Store feature columns
    exclude = {"time", "open", "high", "low", "close", "volume", "target"}
    global FEATURE_COLS
    FEATURE_COLS = [col for col in d.columns if col not in exclude]

    return d


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_feed import fetch_coinbase_candles

    raw  = fetch_coinbase_candles()
    feat = build_features(raw)
    print(f"Shape: {feat.shape}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Target distribution:\n{feat['target'].value_counts()}")
