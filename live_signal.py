"""
live_signal.py — ML All Combined  |  BTC 5-min  |  UP + DOWN  |  Telegram alerts
==================================================================================
Trains XGBoost + CatBoost at startup from the local database, then fires signals
60 seconds before each 5-min candle closes. Retrains daily at midnight UTC.
Sends a Telegram message to your phone on every signal.

Signal conditions:
  UP  : both models predict UP   + ADX > 25 + 15m & 1h EMA both bullish
  DOWN: both models predict DOWN + ADX > 25 + 15m & 1h EMA both bearish + conf>55%
  No session filter — runs 24/7.

Setup:
  1. Create a Telegram bot via @BotFather — copy the token
  2. Message your bot once, then visit:
       https://api.telegram.org/bot<TOKEN>/getUpdates
     Copy your chat_id from the response
  3. Paste token and chat_id below
  4. Run: conda run -n btc_pred python live_signal.py
"""

import os, sys, time, threading, warnings, requests
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import features as feat_mod
from features import build_features
from fetch_store import load_from_db
import xgboost as xgb
from catboost import CatBoostClassifier

# ══════════════════════════════════════════════════════════════════
# TELEGRAM — paste your values here
# ══════════════════════════════════════════════════════════════════
TELEGRAM_TOKEN   = 'YOUR_BOT_TOKEN_HERE'
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID_HERE'

# ══════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════
GRAN        = 300       # 5-min candles = 300 seconds
WARN_SECS   = 0         # fire signal at candle close — enter candle B immediately
ADX_MIN     = 25
TIMING_GUARD = 10       # ignore if less than this many seconds until next signal
SEP  = '═' * 58
SEP2 = '─' * 58

# Shared model state — updated by daily retraining thread
_model_lock = threading.Lock()
_xm = None
_cm = None
_FC = None


# ══════════════════════════════════════════════════════════════════
# TELEGRAM
# ══════════════════════════════════════════════════════════════════
def send_telegram(message: str):
    if TELEGRAM_TOKEN == 'YOUR_BOT_TOKEN_HERE':
        return   # not configured yet
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage',
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'},
            timeout=10)
        if not r.ok:
            print(f'  Telegram send failed: {r.status_code} {r.text[:120]}')
    except Exception as e:
        print(f'  Telegram error: {e}')


# ══════════════════════════════════════════════════════════════════
# DATA FETCHING
# ══════════════════════════════════════════════════════════════════
def _parse(batch):
    # Coinbase returns [time, low, high, open, close, volume]
    df = pd.DataFrame(batch, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_localize(None)
    # Correct column order to standard OHLCV
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    return df.sort_values('time').drop_duplicates('time').reset_index(drop=True)

def fetch_block(n=300):
    end = int(time.time())
    r = requests.get('https://api.exchange.coinbase.com/products/BTC-USD/candles',
                     params={'granularity': GRAN, 'start': end - n * GRAN, 'end': end},
                     timeout=15)
    r.raise_for_status()
    d = r.json()
    return _parse(d) if d and not isinstance(d, dict) else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════
# EMA TREND
# ══════════════════════════════════════════════════════════════════
def ema_flags(df5):
    """Compute 15m and 1h EMA trend flags in one pass. Returns (t15, t1h) as ints."""
    results = {}
    for rule in ('15min', '1h'):
        rs = df5.set_index('time').resample(rule).agg(
            {'open': 'first', 'high': 'max', 'low': 'min',
             'close': 'last', 'volume': 'sum'}).dropna()
        c = rs['close'].astype(float)
        flag = (c.ewm(span=9, adjust=False).mean() >
                c.ewm(span=21, adjust=False).mean()).astype(int).rename('flag')
        merged = pd.merge_asof(df5[['time']], flag.reset_index(),
                               left_on='time', right_on='time',
                               direction='backward')['flag'].fillna(0).astype(int)
        results[rule] = int(merged.iloc[-1])
    return results['15min'], results['1h']


# ══════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════
def train(df_raw):
    print('  Building features...', flush=True)
    data  = build_features(df_raw)
    FC    = feat_mod.FEATURE_COLS
    valid = data[FC + ['target']].dropna()
    X, y  = valid[FC].values, valid['target'].values
    print(f'  Training on {len(X):,} candles  ({y.mean()*100:.1f}% UP rate)', flush=True)

    xm = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric='logloss', random_state=42, verbosity=0)
    xm.fit(X, y)
    cm = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.05,
                            random_seed=42, verbose=0)
    cm.fit(X, y)
    return xm, cm, FC


def retrain_from_db(label='retrain'):
    global _xm, _cm, _FC
    print(f'\n  [{label}] Loading all data from database...', flush=True)
    df_all = load_from_db()
    df_all = df_all.sort_values('time').reset_index(drop=True)
    df_all['time'] = pd.to_datetime(df_all['time'])
    print(f'  [{label}] {len(df_all):,} candles  '
          f'({df_all["time"].iloc[0].date()} → {df_all["time"].iloc[-1].date()})',
          flush=True)
    xm, cm, FC = train(df_all)
    with _model_lock:
        _xm, _cm, _FC = xm, cm, FC
    print(f'  [{label}] Models updated.', flush=True)
    return xm, cm, FC


# ══════════════════════════════════════════════════════════════════
# DAILY RETRAINING THREAD
# ══════════════════════════════════════════════════════════════════
def daily_retrain_loop():
    """Background thread: sleeps until midnight UTC, retrains, repeats."""
    while True:
        now_utc   = pd.Timestamp.now('UTC')
        next_mid  = (now_utc + pd.Timedelta(days=1)).normalize()   # next midnight UTC
        sleep_sec = (next_mid - now_utc).total_seconds()
        print(f'  [retrain] Next retrain at {next_mid.strftime("%Y-%m-%d 00:00 UTC")} '
              f'(in {sleep_sec/3600:.1f}h)', flush=True)
        time.sleep(sleep_sec)
        try:
            retrain_from_db(label='daily retrain')
            send_telegram('🔄 <b>APEX models retrained</b> on full database')
        except Exception as e:
            print(f'  [retrain] ERROR: {e}', flush=True)
            send_telegram(f'⚠️ APEX retrain failed: {e}')


# ══════════════════════════════════════════════════════════════════
# SIGNAL CHECK
# ══════════════════════════════════════════════════════════════════
def check(df_live):
    with _model_lock:
        xm, cm, FC = _xm, _cm, _FC

    try:
        data = build_features(df_live.reset_index(drop=True))
    except Exception as e:
        return {'direction': None, 'reason': f'feature error: {e}'}

    valid = data[FC].dropna()
    if len(valid) == 0:
        return {'direction': None, 'reason': 'not enough data'}

    row = valid.iloc[[-1]]

    xp    = int(xm.predict(row.values)[0])
    cp    = int(cm.predict(row.values)[0])
    xprob = float(xm.predict_proba(row.values)[0][1])
    cprob = float(cm.predict_proba(row.values)[0][1])

    if xp != cp:
        return {'direction': None, 'reason': f'models disagree  XGB:{xp}  CAT:{cp}'}

    adx = float(row['adx_14'].values[0]) if 'adx_14' in row.columns else 0.0
    if adx < ADX_MIN:
        return {'direction': None, 'reason': f'ADX {adx:.1f} < {ADX_MIN}'}

    t15, t1h = ema_flags(df_live)

    price   = float(df_live['close'].iloc[-1])
    up_conf = (xprob + cprob) / 2
    dn_conf = ((1 - xprob) + (1 - cprob)) / 2

    # UP — no confidence filter (53.5% acc, worst streak 7, 0 cap-outs out-of-sample)
    if xp == 1 and t15 == 1 and t1h == 1:
        return {'direction': 'UP', 'xprob': xprob, 'cprob': cprob,
                'conf': up_conf, 'adx': adx, 't15': t15, 't1h': t1h, 'price': price}

    # DOWN — require 55%+ avg confidence (lifts acc from 51% to 61.6% out-of-sample)
    if xp == 0 and t15 == 0 and t1h == 0:
        if dn_conf < 0.55:
            return {'direction': None,
                    'reason': f'DOWN conf {dn_conf*100:.1f}% < 55% threshold'}
        return {'direction': 'DOWN', 'xprob': 1 - xprob, 'cprob': 1 - cprob,
                'conf': dn_conf, 'adx': adx, 't15': t15, 't1h': t1h, 'price': price}

    d = {1: 'UP', 0: 'DOWN'}
    return {'direction': None,
            'reason': f'EMA not aligned  15m:{d[t15]}  1h:{d[t1h]}  model:{d[xp]}'}


# ══════════════════════════════════════════════════════════════════
# TIMING
# ══════════════════════════════════════════════════════════════════
def secs_to_next_signal():
    now  = time.time()
    into = now % GRAN
    wait = GRAN - into   # seconds until next candle close
    # If we just woke up within the guard window of a boundary, skip to the next one
    return wait + GRAN if wait <= TIMING_GUARD else wait

def countdown(secs):
    target = time.time() + secs
    while True:
        rem = target - time.time()
        if rem <= 0:
            break
        m, s = divmod(int(rem), 60)
        print(f'\r  Next check in {m}:{s:02d}   ', end='', flush=True)
        time.sleep(1)
    print('\r' + ' ' * 40, end='\r')


# ══════════════════════════════════════════════════════════════════
# DISPLAY + ALERT
# ══════════════════════════════════════════════════════════════════
def fire_signal(sig: dict):
    direction = sig['direction']
    arrow     = '▲' if direction == 'UP' else '▼'
    emoji     = '🟢' if direction == 'UP' else '🔴'
    now_utc   = pd.Timestamp.now('UTC').strftime('%H:%M UTC')

    t = {1: 'UP', 0: 'DOWN'}
    conf_pct = sig['conf'] * 100

    # Terminal
    print(f'\n{SEP}')
    print(f'  [{now_utc}]  {direction} SIGNAL  {arrow}')
    print(f'  Price      : ${sig["price"]:,.2f}')
    print(f'  Confidence : {conf_pct:.1f}%  (XGB {sig["xprob"]*100:.1f}%  CAT {sig["cprob"]*100:.1f}%)')
    print(f'  ADX        : {sig["adx"]:.1f}')
    print(f'  15m EMA    : {t[sig["t15"]]}   |   1h EMA : {t[sig["t1h"]]}')
    print(f'  Bet next candle {direction}  — ~{WARN_SECS}s to enter')
    print(SEP)

    # Telegram
    msg = (f'{emoji} <b>BTC {direction} Signal</b>\n'
           f'Price : <b>${sig["price"]:,.2f}</b>\n'
           f'Conf  : {conf_pct:.1f}%  (XGB {sig["xprob"]*100:.1f}%  CAT {sig["cprob"]*100:.1f}%)\n'
           f'ADX   : {sig["adx"]:.1f}\n'
           f'15m   : {t[sig["t15"]]}  |  1h : {t[sig["t1h"]]}\n'
           f'<b>Bet next candle {direction}</b>  (~{WARN_SECS}s to enter)')
    send_telegram(msg)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print(f'\n{SEP}')
    print('  BTC ML SIGNAL TERMINAL  —  5-min  |  UP + DOWN  |  24/7')
    tg_status = 'configured' if TELEGRAM_TOKEN != 'YOUR_BOT_TOKEN_HERE' else 'NOT configured'
    print(f'  Telegram  : {tg_status}')
    print(SEP)

    # Initial training from database (fast — no network required)
    print('\n  Loading training data from database...')
    retrain_from_db(label='startup')

    # Start background daily retraining thread
    t = threading.Thread(target=daily_retrain_loop, daemon=True)
    t.start()

    # Send startup notification
    send_telegram('✅ <b>BTC Signal Terminal started</b>\nWatching for UP + DOWN signals 24/7')

    print(f'\n{SEP}')
    print(f'  READY — signals fire {WARN_SECS}s before each candle close')
    print(f'  Retrains daily at midnight UTC')
    print(f'  Ctrl+C to stop')
    print(SEP)

    try:
        while True:
            countdown(secs_to_next_signal())

            now_utc = pd.Timestamp.now('UTC').strftime('%H:%M UTC')

            try:
                df_live = fetch_block(n=300)
            except Exception as e:
                print(f'  [{now_utc}]  fetch error: {e}')
                continue

            if len(df_live) < 60:
                print(f'  [{now_utc}]  not enough live data ({len(df_live)} candles)')
                continue

            sig = check(df_live)

            if sig['direction']:
                fire_signal(sig)
            else:
                print(f'  [{now_utc}]  {sig["reason"]}')

    except KeyboardInterrupt:
        send_telegram('⏹ BTC Signal Terminal stopped')
        print('\n  Stopped.\n')

if __name__ == '__main__':
    main()
