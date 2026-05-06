"""
Shared configuration for BTC 5-min binary direction predictor.
"""

# ── Data settings ─────────────────────────────────────────────────────────────
SYMBOL          = "BTC-USD"          # Coinbase product ID
INTERVAL        = 300                # 5 minutes in seconds
NUM_CANDLES     = 5000               # historical candles to fetch

# ── Feature settings ──────────────────────────────────────────────────────────
SEQUENCE_LEN    = 60                 # LSTM lookback (60 candles = 5 hours)

# ── Backtest settings ─────────────────────────────────────────────────────────
TRAIN_WINDOW    = 2000               # candles per training fold
TEST_WINDOW     = 200                # candles per test fold
STEP            = 200                # slide forward by this many candles

# ── Signal threshold ──────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60          # only signal when model confidence > 60%

# ── Polymarket settings ───────────────────────────────────────────────────────
POLYMARKET_API  = "https://clob.polymarket.com"
