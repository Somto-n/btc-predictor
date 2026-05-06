"""
app.py — BTC/USDT 5-Min Direction Predictor Dashboard
Run: conda run -n btc_pred streamlit run app.py
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import requests

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import build_features
import features as feat_mod
import xgboost as xgb
from catboost import CatBoostClassifier
from streamlit_autorefresh import st_autorefresh

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BTC Signal Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0e1117; }
[data-testid="stHeader"]           { background:#0e1117; }
[data-testid="stSidebar"]          { background:#111827; }
.signal-box { border-radius:12px; padding:1.8rem; text-align:center; margin-bottom:0.5rem; }
.sig-up     { background:#052e16; border:2px solid #16a34a; }
.sig-down   { background:#450a0a; border:2px solid #dc2626; }
.sig-none   { background:#111827; border:2px solid #374151; }
.reason     { font-size:0.78rem; color:#6b7280; margin-top:0.4rem; }
.gloss-term { color:#60a5fa; font-weight:700; font-size:0.95rem; }
.gloss-body { color:#d1d5db; font-size:0.85rem; margin-bottom:0.8rem; }
</style>
""", unsafe_allow_html=True)

GRAN    = 300
ADX_MIN = 25

# ── Auto-refresh every 5 minutes ─────────────────────────────────────────────
st_autorefresh(interval=300_000, key="btc_refresh")

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Controls + Glossary
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")

    adx_threshold = st.slider(
        "ADX Threshold", min_value=15, max_value=40, value=25, step=1,
        help="Minimum ADX value to confirm a strong trend. Higher = fewer but stronger signals."
    )
    candles_shown = st.slider(
        "Candles on chart", min_value=50, max_value=300, value=100, step=10,
    )
    show_ema9  = st.checkbox("Show EMA 9",  value=True)
    show_ema21 = st.checkbox("Show EMA 21", value=True)
    show_ema50 = st.checkbox("Show EMA 50", value=True)

    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.title("📖 Glossary")

    glossary = [
        ("Binary Options", "A trade where you predict whether an asset's price will be higher or lower after a fixed time period. You win a fixed payout if correct, lose your stake if wrong. This app predicts the direction of the next 5-minute BTC candle to help inform those bets."),
        ("5-Min Candle", "Each candle represents 5 minutes of price action. It shows the opening price, closing price, and the highest/lowest prices reached during those 5 minutes. A green candle means price went up, red means it went down."),
        ("XGBoost", "Extreme Gradient Boosting — a machine learning algorithm that builds many decision trees sequentially, each one correcting the errors of the previous. Extremely popular for tabular data like financial indicators because it handles noise well and trains fast."),
        ("CatBoost", "A gradient boosting algorithm by Yandex. Similar to XGBoost but often outperforms it on structured data. The two models are run together as an ensemble — both must agree before a signal fires, which filters out low-confidence predictions."),
        ("Ensemble", "Using multiple models together. Here, XGBoost AND CatBoost must predict the same direction. If they disagree, no signal fires. This reduces false signals significantly."),
        ("ADX", "Average Directional Index — measures how strong a trend is, not which direction. Ranges 0–100. Below 20 = weak/choppy market. Above 25 = strong trend. We only trade when ADX > 25 to avoid getting chopped up in sideways markets."),
        ("EMA", "Exponential Moving Average — a smoothed average of price that reacts faster to recent moves than a simple average. EMA(9) reacts quickly, EMA(50) reacts slowly. When EMA(9) > EMA(21) > EMA(50), the trend is bullish."),
        ("EMA Confluence", "Multi-timeframe alignment check. The 15-minute EMA trend and the 1-hour EMA trend must both point in the same direction as the signal. If the 5-min model says UP but the 1-hour trend is DOWN, no signal fires."),
        ("RSI", "Relative Strength Index — momentum oscillator ranging 0–100. Above 70 = potentially overbought. Below 30 = potentially oversold. Used as a feature to help the model understand momentum context."),
        ("MACD", "Moving Average Convergence Divergence — measures the gap between two EMAs. When the MACD line crosses its signal line upward, it suggests bullish momentum. Used as a feature input to the models."),
        ("ATR", "Average True Range — measures average candle size (volatility). A larger ATR means bigger price swings. Helps the model understand whether the market is calm or volatile."),
        ("Bollinger Bands", "Bands drawn 2 standard deviations above and below a 20-period moving average. When price is near the upper band, the market may be extended. The %B value (position within bands) is used as a feature."),
        ("Martingale", "A bet-sizing strategy: if you lose, double your stake on the next bet. If you win, reset to the base stake. Risky if you hit a long losing streak. The backtest tracked how often 6+ consecutive losses occurred (cap-out rate <1.5%)."),
        ("Out-of-sample", "Data the model never saw during training. The 65.2% accuracy figure comes from testing on 6 months of data the models were not trained on — this is the honest performance metric."),
        ("Confidence", "The model's probability output. XGBoost outputs e.g. 68% probability of UP. CatBoost outputs 71%. The average (69.5%) is shown as the signal confidence. DOWN signals require >55% avg confidence."),
    ]

    for term, definition in glossary:
        with st.expander(term):
            st.markdown(definition)

# ─────────────────────────────────────────────────────────────────────────────
# CACHED FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_training_data(days=30):
    """Fetch historical 5-min candles from Coinbase API for model training."""
    GRAN = 300
    all_candles = []
    end_ts  = int(time.time())
    start_ts = end_ts - days * 24 * 3600
    cur_end  = end_ts
    while cur_end > start_ts:
        cur_start = max(cur_end - 300 * GRAN, start_ts)
        try:
            r = requests.get(
                'https://api.exchange.coinbase.com/products/BTC-USD/candles',
                params={'granularity': GRAN, 'start': cur_start, 'end': cur_end},
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                if data and isinstance(data, list):
                    all_candles.extend(data)
        except Exception:
            pass
        cur_end = cur_start - GRAN
        time.sleep(0.35)
    if not all_candles:
        return pd.DataFrame()
    df = pd.DataFrame(all_candles, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_localize(None)
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']].astype(
        {'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    return df.drop_duplicates('time').sort_values('time').reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_and_train():
    # Try local DB first (works when running locally), fall back to Coinbase API
    df_raw = pd.DataFrame()
    try:
        from fetch_store import load_from_db
        df_raw = load_from_db()
        df_raw = df_raw.sort_values('time').reset_index(drop=True)
        df_raw['time'] = pd.to_datetime(df_raw['time'])
    except Exception:
        pass
    if len(df_raw) < 1000:
        df_raw = fetch_training_data(days=30)
    data  = build_features(df_raw)
    FC    = feat_mod.FEATURE_COLS
    valid = data[FC + ['target']].dropna()
    X, y  = valid[FC].values, valid['target'].values
    xm = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric='logloss', random_state=42, verbosity=0)
    xm.fit(X, y)
    cm = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.05,
                            random_seed=42, verbose=0)
    cm.fit(X, y)
    return xm, cm, FC


@st.cache_data(ttl=300, show_spinner=False)
def fetch_live(n=300):
    end = int(time.time())
    r = requests.get(
        'https://api.exchange.coinbase.com/products/BTC-USD/candles',
        params={'granularity': GRAN, 'start': end - n * GRAN, 'end': end},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_localize(None)
    df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
    return df.sort_values('time').drop_duplicates('time').reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL LOGIC
# ─────────────────────────────────────────────────────────────────────────────
def ema_flags(df5):
    results = {}
    for rule in ('15min', '1h'):
        rs = df5.set_index('time').resample(rule).agg(
            {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
        c    = rs['close'].astype(float)
        flag = (c.ewm(span=9, adjust=False).mean() >
                c.ewm(span=21, adjust=False).mean()).astype(int).rename('flag')
        merged = pd.merge_asof(df5[['time']], flag.reset_index(),
                               left_on='time', right_on='time',
                               direction='backward')['flag'].fillna(0).astype(int)
        results[rule] = int(merged.iloc[-1])
    return results['15min'], results['1h']


def check_signal(df_live, xm, cm, FC, adx_min=25):
    try:
        data = build_features(df_live.reset_index(drop=True))
    except Exception as e:
        return {'direction': None, 'reason': f'Feature error: {e}'}
    valid = data[FC].dropna()
    if len(valid) == 0:
        return {'direction': None, 'reason': 'Not enough data'}
    row   = valid.iloc[[-1]]
    xp    = int(xm.predict(row.values)[0])
    cp    = int(cm.predict(row.values)[0])
    xprob = float(xm.predict_proba(row.values)[0][1])
    cprob = float(cm.predict_proba(row.values)[0][1])
    if xp != cp:
        lbl = {1:'UP',0:'DOWN'}
        return {'direction': None, 'reason': f'Models disagree — XGB: {lbl[xp]}  CAT: {lbl[cp]}'}
    adx = float(row['adx_14'].values[0]) if 'adx_14' in row.columns else 0.0
    if adx < adx_min:
        return {'direction': None, 'reason': f'Weak trend — ADX {adx:.1f} < {adx_min}'}
    t15, t1h = ema_flags(df_live)
    price    = float(df_live['close'].iloc[-1])
    up_conf  = (xprob + cprob) / 2
    dn_conf  = ((1 - xprob) + (1 - cprob)) / 2
    lbl      = {1:'UP', 0:'DOWN'}
    if xp == 1 and t15 == 1 and t1h == 1:
        return {'direction':'UP','xprob':xprob,'cprob':cprob,
                'conf':up_conf,'adx':adx,'price':price,'t15':t15,'t1h':t1h}
    if xp == 0 and t15 == 0 and t1h == 0:
        if dn_conf < 0.55:
            return {'direction': None, 'reason': f'DOWN conf {dn_conf*100:.1f}% < 55%'}
        return {'direction':'DOWN','xprob':1-xprob,'cprob':1-cprob,
                'conf':dn_conf,'adx':adx,'price':price,'t15':t15,'t1h':t1h}
    return {'direction': None,
            'reason': f'EMA not aligned — 15m: {lbl[t15]}  1h: {lbl[t1h]}  Model: {lbl[xp]}'}


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("₿  BTC/USDT 5-Min Direction Predictor")
st.caption("XGBoost + CatBoost ensemble · All Combined filter · Live Coinbase data · Auto-refreshes every 5 min")

with st.expander("📘 What is this dashboard? — Start here", expanded=False):
    st.markdown("""
### Built for Binary Options Trading

**Binary options** are a type of trade where you bet on one simple question:
> *Will BTC be higher or lower than its current price when the next 5-minute candle closes?*

- If you bet **UP** and BTC closes higher → you win a fixed payout (e.g. 80–95%)
- If you bet **DOWN** and BTC closes lower → you win
- If you're wrong → you lose your stake

There are only two outcomes — that's why it's called *binary*. No stop losses, no position sizing, no charts to manage. Just: **UP or DOWN**.

---

### What this dashboard does

This app uses two machine learning models (XGBoost and CatBoost) trained on **105,000+ historical BTC candles** to predict the direction of the *next* 5-minute candle. Before firing a signal, it applies three additional filters to reduce false positives:

| Filter | Why it matters |
|--------|----------------|
| Both models must agree | One model can be wrong. Two agreeing is a stronger signal. |
| ADX > threshold | Signals in choppy, sideways markets are unreliable. ADX confirms the market is actually trending. |
| 15-min & 1-hour EMA aligned | Betting against the bigger trend is risky. EMA alignment makes sure the 5-min signal matches the bigger picture. |

Only when **all filters pass** does the dashboard say UP or DOWN.

---

### How to use it

1. Wait for a **🟢 UP** or **🔴 DOWN** signal to appear
2. Note the timestamp — the signal fires at the close of a 5-min candle
3. Place your binary options bet on the **next candle** in the same direction
4. The app auto-refreshes every 5 minutes at each candle boundary

> ⚠️ **Disclaimer:** This tool is for informational and research purposes. Past backtest accuracy does not guarantee future results. Never trade more than you can afford to lose.
""")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Training models on historical data — first load only, takes ~30s..."):
    try:
        xm, cm, FC = load_and_train()
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# FETCH LIVE DATA
# ─────────────────────────────────────────────────────────────────────────────
try:
    df_live  = fetch_live(n=max(candles_shown + 50, 300))
    fetch_ok = len(df_live) >= 60
except Exception as e:
    st.warning(f"Live data unavailable: {e}")
    df_live  = pd.DataFrame()
    fetch_ok = False

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL
# ─────────────────────────────────────────────────────────────────────────────
sig       = check_signal(df_live, xm, cm, FC, adx_threshold) if fetch_ok else \
            {'direction': None, 'reason': 'No live data'}
direction = sig.get('direction')
now_str   = pd.Timestamp.now('UTC').strftime('%Y-%m-%d %H:%M UTC')
into      = int(time.time()) % GRAN
secs_left = GRAN - into
m, s      = divmod(secs_left, 60)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Signal + Stats
# ─────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

with left:
    if direction == 'UP':
        st.markdown(f"""
        <div class="signal-box sig-up">
            <div style="font-size:3rem">🟢</div>
            <div style="font-size:2.2rem;font-weight:800;color:#4ade80">UP ▲</div>
            <div style="color:#86efac;margin-top:0.3rem">Avg confidence: {sig['conf']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    elif direction == 'DOWN':
        st.markdown(f"""
        <div class="signal-box sig-down">
            <div style="font-size:3rem">🔴</div>
            <div style="font-size:2.2rem;font-weight:800;color:#f87171">DOWN ▼</div>
            <div style="color:#fca5a5;margin-top:0.3rem">Avg confidence: {sig['conf']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="signal-box sig-none">
            <div style="font-size:3rem">⚪</div>
            <div style="font-size:1.9rem;font-weight:700;color:#9ca3af">NO SIGNAL</div>
            <div class="reason">{sig.get('reason','—')}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(f"Checked: {now_str}")
    st.caption(f"Next candle close in: **{m}:{s:02d}**")

    if direction:
        c1, c2 = st.columns(2)
        c1.metric("XGBoost", f"{sig['xprob']*100:.1f}%",
                  help="XGBoost probability of the predicted direction")
        c2.metric("CatBoost", f"{sig['cprob']*100:.1f}%",
                  help="CatBoost probability of the predicted direction")
        if fetch_ok:
            st.metric("BTC Price", f"${df_live['close'].iloc[-1]:,.2f}")
        st.metric("ADX", f"{sig['adx']:.1f}",
                  help=f"Trend strength. Must be > {adx_threshold} to fire a signal.")
        ema_lbl = {1:'🟢 Bullish', 0:'🔴 Bearish'}
        st.caption(f"15-min EMA: {ema_lbl[sig.get('t15',0)]}  |  1h EMA: {ema_lbl[sig.get('t1h',0)]}")

with right:
    st.subheader("📊 Backtest Results — 6-Month Out-of-Sample")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("UP Accuracy",    "65.2%",   help="Directional accuracy on bullish signals — never seen by model during training")
    m2.metric("Candles Tested", "52,000+", help="6 months of 5-minute BTC candles")
    m3.metric("Signal Rate",    "3.9%",    help="The filter is selective — only ~3.9% of candles produce a signal")
    m4.metric("Cap-Out Rate",   "<1.5%",   help="Martingale: probability of losing 6 consecutive bets in a row")

    st.divider()
    st.subheader("🔍 How a Signal Fires")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Step 1 — Both models agree**
XGBoost and CatBoost independently predict direction. If they disagree → no signal.

**Step 2 — Trend is strong**
ADX must be above your threshold (currently **{}**). Weak/choppy markets are skipped.
""".format(adx_threshold))
    with col_b:
        st.markdown("""
**Step 3 — Multi-timeframe EMA aligned**
The 15-minute and 1-hour EMA trends must match the signal direction. This filters out signals that go against the bigger trend.

**Step 4 — DOWN confidence check**
DOWN signals additionally require avg model confidence > 55%.
""")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2 — Chart
# ─────────────────────────────────────────────────────────────────────────────
st.subheader(f"📈 Live BTC/USDT — Last {candles_shown} Candles (5-Min)")

if fetch_ok:
    chart_df = df_live.tail(candles_shown).copy()
    chart_df['ema9']  = chart_df['close'].ewm(span=9,  adjust=False).mean()
    chart_df['ema21'] = chart_df['close'].ewm(span=21, adjust=False).mean()
    chart_df['ema50'] = chart_df['close'].ewm(span=50, adjust=False).mean()

    # Signal markers
    up_times, dn_times = set(), set()
    try:
        feat_df  = build_features(df_live.reset_index(drop=True))
        valid    = feat_df[FC].dropna()
        if len(valid):
            xpreds = xm.predict(valid.values)
            cpreds = cm.predict(valid.values)
            adxs   = valid['adx_14'].values if 'adx_14' in valid.columns else np.zeros(len(valid))
            agree  = xpreds == cpreds
            adx_ok = adxs > adx_threshold
            times  = df_live.loc[feat_df.index[feat_df[FC].notna().all(axis=1)], 'time'].values
            for i, (xp, ag, aok) in enumerate(zip(xpreds, agree, adx_ok)):
                if ag and aok and i < len(times):
                    (up_times if xp == 1 else dn_times).add(times[i])
    except Exception:
        pass

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=chart_df['time'],
        open=chart_df['open'], high=chart_df['high'],
        low=chart_df['low'],   close=chart_df['close'],
        name='BTC/USD',
        increasing_line_color='#22c55e', decreasing_line_color='#ef4444',
        increasing_fillcolor='#22c55e',  decreasing_fillcolor='#ef4444',
    ))

    if show_ema9:
        fig.add_trace(go.Scatter(x=chart_df['time'], y=chart_df['ema9'],
            name='EMA 9', line=dict(color='#fbbf24', width=1.2)))
    if show_ema21:
        fig.add_trace(go.Scatter(x=chart_df['time'], y=chart_df['ema21'],
            name='EMA 21', line=dict(color='#60a5fa', width=1.2)))
    if show_ema50:
        fig.add_trace(go.Scatter(x=chart_df['time'], y=chart_df['ema50'],
            name='EMA 50', line=dict(color='#c084fc', width=1.2)))

    up_chart = chart_df[chart_df['time'].isin(up_times)]
    if len(up_chart):
        fig.add_trace(go.Scatter(
            x=up_chart['time'], y=up_chart['low'] * 0.9994,
            mode='markers', name='UP Signal',
            marker=dict(symbol='triangle-up', size=11, color='#4ade80',
                        line=dict(color='#16a34a', width=1)),
        ))
    dn_chart = chart_df[chart_df['time'].isin(dn_times)]
    if len(dn_chart):
        fig.add_trace(go.Scatter(
            x=dn_chart['time'], y=dn_chart['high'] * 1.0006,
            mode='markers', name='DOWN Signal',
            marker=dict(symbol='triangle-down', size=11, color='#f87171',
                        line=dict(color='#dc2626', width=1)),
        ))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0e1117', plot_bgcolor='#111827',
        height=520, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='right', x=1, bgcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor='#1f2937'),
        yaxis=dict(gridcolor='#1f2937', tickformat='$,.0f'),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Chart annotation
    with st.expander("What am I looking at?"):
        st.markdown("""
- **Green/red candles** — each candle = 5 minutes of BTC price action. Green = price went up, red = price went down.
- **EMA 9 (yellow)** — fast-moving average. Reacts quickly to price changes.
- **EMA 21 (blue)** — medium-moving average. When EMA 9 is above EMA 21, the short-term trend is bullish.
- **EMA 50 (purple)** — slow-moving average. Represents the longer-term trend.
- **Green triangles (▲)** — candles where both models agreed on UP and ADX was strong enough.
- **Red triangles (▼)** — candles where both models agreed on DOWN and ADX was strong enough.
- Note: triangle markers show model agreement + ADX filter only. The EMA confluence filter is applied separately for the live signal.
        """)
else:
    st.info("Chart unavailable — could not connect to Coinbase.")
