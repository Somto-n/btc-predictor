"""
Extended Backtest — 6+ months of BTC 5-min data from CryptoCompare
-------------------------------------------------------------------
Settings:  Base stake $5 | Bankroll $500 | Martingale max 6 steps

Data source: CryptoCompare histominute API (free, no key required)
             Closely mirrors Chainlink BTC-USD (same underlying exchanges)
             Resampled from 1-min to 5-min OHLCV.

Models: XGBoost + CatBoost only (LSTM skipped for speed on large dataset)
Walk-forward with larger windows to suit the dataset size.
"""

import os, sys, time, warnings, requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, os.path.dirname(__file__))

from features import build_features, FEATURE_COLS
from backtest import _max_streak, _streak_percentiles

# ── Settings ──────────────────────────────────────────────────────────────────
BASE_STAKE   = 5.0
BANKROLL     = 500.0
MULTIPLIER   = 2.0
MAX_STEPS    = 6          # user's hard cap both directions
MONTHS_BACK  = 6          # how many months of data to fetch
TRAIN_WINDOW = 2000       # ~7 days of 5-min candles
TEST_WINDOW  = 500
STEP         = 500
CONF_THRESH  = 0.60       # ensemble confidence threshold


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FETCH DATA — CryptoCompare histominute → resample to 5-min
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_coinbase_large(months: int = 6, granularity: int = 300) -> pd.DataFrame:
    """
    Paginate Coinbase API backwards to collect `months` of 5-min OHLCV.
    Coinbase allows max 300 candles per request.
    """
    symbol      = "BTC-USD"
    url         = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    target_secs = months * 30 * 24 * 3600
    total_need  = target_secs // granularity
    max_per_req = 300

    all_rows = []
    end_ts   = int(time.time())
    calls    = 0

    print(f"Fetching ~{months} months of BTC-USD {granularity//60}-min candles "
          f"from Coinbase...")
    print(f"Target: ~{total_need:,} candles  "
          f"(~{total_need // max_per_req} API calls)")

    while len(all_rows) < total_need:
        start_ts = end_ts - (max_per_req * granularity)
        params   = {"granularity": granularity,
                    "start": start_ts, "end": end_ts}
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            print(f"\n  API error: {e} — retrying in 3s")
            time.sleep(3)
            continue

        if not batch or isinstance(batch, dict):
            break

        all_rows.extend(batch)
        end_ts  = start_ts
        calls  += 1

        if calls % 20 == 0:
            oldest = pd.to_datetime(batch[-1][0], unit="s").date()
            print(f"  {calls:>4} calls | {len(all_rows):>7,} candles | "
                  f"oldest: {oldest}")

        time.sleep(0.2)

    df = pd.DataFrame(all_rows,
                      columns=["time","low","high","open","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = (df.sort_values("time")
            .drop_duplicates("time")
            .reset_index(drop=True))
    df = df[df["close"] > 0].reset_index(drop=True)

    print(f"\n  Fetched {len(df):,} candles  "
          f"({df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()})")
    return df


raw5 = fetch_coinbase_large(months=MONTHS_BACK)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\nBuilding features...")
data = build_features(raw5)
from features import FEATURE_COLS
print(f"  {len(data):,} candles  |  {len(FEATURE_COLS)} features")
print(f"  Target: UP={data['target'].mean()*100:.1f}%  "
      f"DOWN={(1-data['target'].mean())*100:.1f}%\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MULTI-TIMEFRAME TREND (15-min and 1-hour)
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing multi-timeframe trends...")

# Build 15-min and 1-hour from the same raw data
raw5_idx = raw5.set_index("time")

raw15 = raw5_idx.resample("15min").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum"
}).dropna().reset_index()

raw1h = raw5_idx.resample("1h").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum"
}).dropna().reset_index()

def ema_trend(df):
    c = df["close"].astype(float)
    return pd.Series((c.ewm(span=9,  adjust=False).mean() >
                      c.ewm(span=21, adjust=False).mean()).astype(int).values,
                     index=df["time"])

trend_15m = ema_trend(raw15)
trend_1h  = ema_trend(raw1h)

data["trend_15m"] = pd.merge_asof(
    data[["time"]], trend_15m.rename("t").reset_index(),
    left_on="time", right_on="time", direction="backward"
)["t"].fillna(0).astype(int).values

data["trend_1h"] = pd.merge_asof(
    data[["time"]], trend_1h.rename("t").reset_index(),
    left_on="time", right_on="time", direction="backward"
)["t"].fillna(0).astype(int).values

data["hour_utc"] = pd.to_datetime(data["time"]).dt.hour


# ═══════════════════════════════════════════════════════════════════════════════
# 4. WALK-FORWARD — XGBoost + CatBoost only
# ═══════════════════════════════════════════════════════════════════════════════
import xgboost as xgb
from catboost import CatBoostClassifier
from backtest import _make_sequences

print("Running walk-forward backtest (XGBoost + CatBoost)...")

X = data[FEATURE_COLS].values
y = data["target"].values
n = len(data)

records = []

fold = 0
for start in range(0, n - TRAIN_WINDOW - TEST_WINDOW, STEP):
    tr_end = start + TRAIN_WINDOW
    te_end = tr_end + TEST_WINDOW

    Xtr, ytr = X[start:tr_end], y[start:tr_end]
    Xte, yte = X[tr_end:te_end], y[tr_end:te_end]

    meta = data.iloc[tr_end:te_end][
        ["time", "adx_14", "hour_utc", "trend_15m", "trend_1h", "target"]
    ].copy()

    # XGBoost
    xgb_m = xgb.XGBClassifier(n_estimators=200, max_depth=5,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, eval_metric="logloss",
                                random_state=42, verbosity=0)
    xgb_m.fit(Xtr, ytr)
    xgb_probs = xgb_m.predict_proba(Xte)[:, 1]
    xgb_preds = xgb_m.predict(Xte)

    # CatBoost
    cat_m = CatBoostClassifier(iterations=200, depth=5, learning_rate=0.05,
                                random_seed=42, verbose=0)
    cat_m.fit(Xtr, ytr)
    cat_probs = cat_m.predict_proba(Xte)[:, 1]
    cat_preds = cat_m.predict(Xte).astype(int)

    for i, idx in enumerate(meta.index):
        records.append({
            "time":      meta.at[idx, "time"],
            "actual":    int(meta.at[idx, "target"]),
            "adx_14":    meta.at[idx, "adx_14"],
            "hour_utc":  int(meta.at[idx, "hour_utc"]),
            "trend_15m": int(meta.at[idx, "trend_15m"]),
            "trend_1h":  int(meta.at[idx, "trend_1h"]),
            "xgb_pred":  int(xgb_preds[i]),
            "xgb_prob":  float(xgb_probs[i]),
            "cat_pred":  int(cat_preds[i]),
            "cat_prob":  float(cat_probs[i]),
        })

    fold += 1
    print(f"  Fold {fold:>3}  test candles so far: {fold * TEST_WINDOW:>7,}", end="\r")

print(f"\n  Done — {fold} folds  |  {len(records):,} test candles")
df = pd.DataFrame(records)
actuals = df["actual"].values


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ALL COMBINED FILTER
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_2model(df):
    """Both XGBoost and CatBoost must agree on direction."""
    sig = np.full(len(df), -1, dtype=int)
    agree_up   = (df["xgb_pred"].values == 1) & (df["cat_pred"].values == 1)
    agree_down = (df["xgb_pred"].values == 0) & (df["cat_pred"].values == 0)
    sig[agree_up]   = 1
    sig[agree_down] = 0
    return sig

sig = ensemble_2model(df)

# Regime filter
sig[df["adx_14"].values < 25] = -1
# Time filter (7-21 UTC active sessions)
sig[(df["hour_utc"].values < 7) | (df["hour_utc"].values >= 21)] = -1
# Multi-TF confluence
t15 = df["trend_15m"].values
t1h = df["trend_1h"].values
sig[(sig == 1) & ~((t15 == 1) & (t1h == 1))] = -1
sig[(sig == 0) & ~((t15 == 0) & (t1h == 0))] = -1

n_sig    = (sig != -1).sum()
n_up     = (sig == 1).sum()
n_down   = (sig == 0).sum()
n_test   = len(df)

print(f"\nAll Combined signals: {n_sig}  "
      f"({n_sig/n_test*100:.2f}% of {n_test:,} candles)")
print(f"  UP: {n_up}  DOWN: {n_down}")

if n_sig == 0:
    print("No signals generated — check data or filter parameters.")
    import sys; sys.exit(0)

if n_sig > 0:
    mask    = sig != -1
    preds_f = sig[mask]
    acts_f  = actuals[mask]
    acc     = (preds_f == acts_f).mean()
    up_acc  = (preds_f[preds_f==1] == acts_f[preds_f==1]).mean() if n_up > 0 else 0
    dn_acc  = (preds_f[preds_f==0] == acts_f[preds_f==0]).mean() if n_down > 0 else 0
    print(f"  Accuracy: {acc*100:.1f}%  UP: {up_acc*100:.1f}%  DOWN: {dn_acc*100:.1f}%\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CONSECUTIVE CANDLE WIN PROBABILITY AT EACH STEP
# ═══════════════════════════════════════════════════════════════════════════════

def consecutive_win_probs(sig, actuals, direction, max_steps, look=8):
    label = "UP" if direction == 1 else "DOWN"
    n     = len(sig)
    sequences = []

    for idx in range(n):
        if sig[idx] != direction:
            continue
        outcomes = []
        for step in range(look):
            ci = idx + step
            if ci >= n:
                break
            outcomes.append(int(actuals[ci] == direction))
        sequences.append(outcomes)

    print(f"\n{'─'*65}")
    print(f"{label} DIRECTION — {len(sequences)} signals")
    print(f"{'─'*65}")
    print(f"  {'Step':>5}  {'Eligible':>9}  {'Won here':>9}  "
          f"{'Win%':>7}  {'Cumul%':>8}  {'Still losing':>13}")
    print(f"  {'-'*60}")

    eligible  = sequences
    total_seq = len(sequences)
    cumul_won = 0
    step_data = []

    for step in range(max_steps):
        at_step  = [s for s in eligible if len(s) > step]
        n_elig   = len(at_step)
        if n_elig == 0:
            break
        won_here = [s for s in at_step if s[step] == 1]
        n_won    = len(won_here)
        win_pct  = n_won / n_elig * 100
        cumul_won += n_won
        c_pct    = cumul_won / total_seq * 100
        lost     = [s for s in at_step if s[step] == 0]

        bar = "█" * int(win_pct / 4)
        print(f"  {step+1:>5}  {n_elig:>9,}  {n_won:>9,}  "
              f"{win_pct:>6.1f}%  {c_pct:>7.1f}%  {len(lost):>13,}  {bar}")
        step_data.append({"step": step+1, "eligible": n_elig,
                           "won": n_won, "win_pct": win_pct, "cumul_pct": c_pct})
        eligible = lost

    never = len(eligible)
    print(f"\n  Never recovered within {max_steps} steps: "
          f"{never} ({never/total_seq*100:.2f}%)")
    return step_data, sequences


probs_up,   seqs_up   = consecutive_win_probs(sig, actuals, 1, MAX_STEPS)
probs_down, seqs_down = consecutive_win_probs(sig, actuals, 0, MAX_STEPS)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. MARTINGALE SIMULATION — $5 base, $500 bankroll, 6 steps
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_martingale(sig, actuals, direction,
                        base=BASE_STAKE, bankroll=BANKROLL,
                        max_steps=MAX_STEPS, mult=MULTIPLIER):
    label    = "UP" if direction == 1 else "DOWN"
    bk       = bankroll
    equity   = [bk]
    peak     = bk
    max_dd   = 0.0
    n        = len(sig)

    in_seq         = False
    cur_step       = 0
    accum_loss     = 0.0
    which_won      = []
    capped         = []
    all_stakes     = []
    consec_loss    = 0
    max_consec     = 0
    seq_outcomes   = []       # "won" or "capped" for every sequence
    consec_caps    = 0        # current run of consecutive cap-outs
    max_consec_cap = 0        # worst back-to-back cap-out run

    i = 0
    while i < n:
        if not in_seq and sig[i] == direction:
            in_seq     = True
            cur_step   = 0
            accum_loss = 0.0

        if in_seq:
            stake = min(base * (mult ** cur_step), bk)
            won   = (actuals[i] == direction)
            all_stakes.append(stake)

            if won:
                bk += stake
                which_won.append(cur_step)
                consec_loss  = 0
                consec_caps  = 0          # winning resets cap run
                seq_outcomes.append("won")
                in_seq    = False
                cur_step  = 0
                accum_loss = 0.0
            else:
                bk        -= stake
                accum_loss += stake
                consec_loss += 1
                max_consec   = max(max_consec, consec_loss)

                if cur_step >= max_steps - 1:
                    capped.append(accum_loss)
                    seq_outcomes.append("capped")
                    consec_caps    += 1
                    max_consec_cap  = max(max_consec_cap, consec_caps)
                    consec_loss = 0
                    in_seq    = False
                    cur_step  = 0
                    accum_loss = 0.0
                else:
                    cur_step += 1

            equity.append(bk)
            peak   = max(peak, bk)
            max_dd = max(max_dd, peak - bk)
            if bk <= 0:
                break
        else:
            equity.append(bk)

        i += 1

    # Consecutive cap-out distribution
    cap_runs, run = [], 0
    for o in seq_outcomes:
        if o == "capped":
            run += 1
        else:
            if run: cap_runs.append(run)
            run = 0
    if run: cap_runs.append(run)

    return {
        "direction":      label,
        "equity":         equity,
        "final":          bk,
        "profit":         bk - bankroll,
        "roi":            (bk - bankroll) / bankroll * 100,
        "max_dd":         max_dd,
        "which_won":      which_won,
        "capped":         capped,
        "n_seq":          len(which_won) + len(capped),
        "n_capped":       len(capped),
        "max_consec":     max_consec,
        "max_consec_cap": max_consec_cap,
        "cap_runs":       cap_runs,
        "seq_outcomes":   seq_outcomes,
        "all_stakes":     all_stakes,
    }


# ── No-ceiling streak finder ──────────────────────────────────────────────────
def nocap_streak_analysis(sig, actuals, direction, max_look=60):
    """
    For every signal, count consecutive candles in `direction` until first win.
    No ceiling — finds the TRUE maximum losing streak.
    """
    label     = "UP" if direction == 1 else "DOWN"
    n         = len(sig)
    losses_before_win = []   # how many losses before the first win
    never_won         = 0

    for idx in range(n):
        if sig[idx] != direction:
            continue
        losses = 0
        found_win = False
        for step in range(max_look):
            ci = idx + step
            if ci >= n:
                break
            if actuals[ci] == direction:
                found_win = True
                break
            losses += 1
        if found_win:
            losses_before_win.append(losses)
        else:
            never_won += 1

    arr = np.array(losses_before_win) if losses_before_win else np.array([0])

    print(f"\n{'═'*68}")
    print(f"NO-CAP STREAK ANALYSIS — {label}  ({len(losses_before_win)+never_won} signals)")
    print(f"{'═'*68}")
    print(f"  Signals that eventually won:  {len(losses_before_win)}")
    print(f"  Never won within {max_look} candles:   {never_won}")
    print(f"\n  Losses before first win:")
    print(f"  {'─'*45}")
    print(f"  Min losses before win:    {arr.min():.0f}")
    print(f"  Average losses before win:{arr.mean():.2f}")
    print(f"  Median:                   {np.median(arr):.1f}")
    print(f"  p90:                      {np.percentile(arr,90):.0f}")
    print(f"  p95:                      {np.percentile(arr,95):.0f}")
    print(f"  p99:                      {np.percentile(arr,99):.0f}")
    print(f"  p99.9:                    {np.percentile(arr,99.9):.0f}")
    print(f"  MAX (absolute worst):     {arr.max():.0f}  ← true uncapped streak")

    print(f"\n  Distribution of losses before win:")
    for k in range(int(arr.max()) + 1):
        cnt = int((arr == k).sum())
        if cnt == 0: continue
        pct = cnt / len(arr) * 100
        bar = "█" * max(1, int(pct / 2))
        print(f"    {k:>3} losses → {cnt:>4}x  ({pct:>5.1f}%)  {bar}")

    print(f"\n  Bankroll needed if no cap (Martingale to deepest streak):")
    worst = int(arr.max())
    total_exp = sum(BASE_STAKE * (MULTIPLIER**s) for s in range(worst + 1))
    print(f"  Worst streak = {worst} losses → need ${total_exp:,.0f} total exposure")
    print(f"  (vs $500 bankroll — {'SAFE' if total_exp<=500 else f'NEEDS ${total_exp-500:,.0f} MORE'})")

    return arr, losses_before_win


sim_up   = simulate_martingale(sig, actuals, 1)
sim_down = simulate_martingale(sig, actuals, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PRINT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*68}")
print(f"MARTINGALE RESULTS  |  Base ${BASE_STAKE} | Bankroll ${BANKROLL} | {MAX_STEPS} steps")
print(f"{'='*68}")

for sim in [sim_up, sim_down]:
    cap_pct  = sim["n_capped"] / max(1, sim["n_seq"]) * 100
    won_pct  = len(sim["which_won"]) / max(1, sim["n_seq"]) * 100
    print(f"\n  {sim['direction']} direction:")
    print(f"  {'─'*50}")
    print(f"  Starting bankroll:     ${BANKROLL:>8,.2f}")
    print(f"  Final bankroll:        ${sim['final']:>8,.2f}")
    print(f"  Total profit:          ${sim['profit']:>8,.2f}  ({sim['roi']:+.1f}% ROI)")
    print(f"  Max drawdown:          ${sim['max_dd']:>8,.2f}")
    print(f"  Max consecutive loss:  {sim['max_consec']:>8}")
    print(f"  Total sequences:       {sim['n_seq']:>8,}")
    print(f"  Won within {MAX_STEPS} steps:   {len(sim['which_won']):>8,}  ({won_pct:.1f}%)")
    print(f"  Hit cap (lost):        {sim['n_capped']:>8,}  ({cap_pct:.1f}%)")
    if sim["capped"]:
        print(f"  Avg cap loss:          ${np.mean(sim['capped']):>8,.2f}")
        print(f"  Worst cap loss:        ${np.max(sim['capped']):>8,.2f}")

print(f"\n{'─'*68}")
print("WHICH STEP THE WIN COMES ON")
print(f"{'─'*68}")
for sim in [sim_up, sim_down]:
    print(f"\n  {sim['direction']}  ({len(sim['which_won'])} won sequences):")
    if not sim["which_won"]: continue
    for s in range(MAX_STEPS):
        count = sim["which_won"].count(s)
        pct   = count / len(sim["which_won"]) * 100 if sim["which_won"] else 0
        stake = BASE_STAKE * (MULTIPLIER ** s)
        bar   = "█" * max(1, int(pct / 3))
        print(f"    Step {s+1} (${stake:>5.0f}): {count:>5,}x  ({pct:>5.1f}%)  {bar}")

print(f"\n{'─'*68}")
print("BACK-TO-BACK CAP-OUT ANALYSIS (consecutive losing sequences)")
print(f"{'─'*68}")
for sim in [sim_up, sim_down]:
    runs = sim["cap_runs"]
    print(f"\n  {sim['direction']}:")
    print(f"  Total cap-outs:            {sim['n_capped']}")
    print(f"  Max consecutive cap-outs:  {sim['max_consec_cap']}")
    if runs:
        print(f"  Cap-out run distribution:")
        for k in range(1, max(runs) + 1):
            cnt = runs.count(k)
            if cnt == 0: continue
            loss = k * (BASE_STAKE * (MULTIPLIER**MAX_STEPS - 1))
            print(f"    {k} in a row → {cnt}x  (total loss: ${loss:,.0f})")
    else:
        print(f"  No cap-out runs recorded.")

print(f"\n{'─'*68}")
print("WORST-CASE EXPOSURE TABLE  ($5 base, 6 steps, $500 bankroll)")
print(f"{'─'*68}")
total = 0.0
print(f"  {'Step':>5}  {'Bet':>7}  {'Total at risk':>14}  {'Net if win':>11}  {'% of bankroll':>14}")
print(f"  {'-'*58}")
for s in range(MAX_STEPS):
    bet    = BASE_STAKE * (MULTIPLIER ** s)
    total += bet
    net    = bet - (total - bet)
    bk_pct = total / BANKROLL * 100
    ok     = "✓" if total <= BANKROLL else "✗"
    print(f"  {s+1:>5}  ${bet:>6.0f}  ${total:>13.0f}  ${net:>+10.0f}  "
          f"{bk_pct:>13.1f}%  {ok}")


nocap_up,   _ = nocap_streak_analysis(sig, actuals, 1,  max_look=60)
nocap_down, _ = nocap_streak_analysis(sig, actuals, 0,  max_look=60)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 15))
fig.suptitle(
    f"Extended Backtest — {MONTHS_BACK}-Month BTC 5-Min  |  "
    f"${BASE_STAKE} base · ${BANKROLL} bankroll · {MAX_STEPS}-step Martingale · "
    f"All Combined Signals",
    fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)

# Plot 1 & 2: Equity curves
for ax_i, sim, col in [(gs[0,0], sim_up, "steelblue"),
                        (gs[0,1], sim_down, "coral")]:
    ax = fig.add_subplot(ax_i)
    ax.plot(sim["equity"], color=col, linewidth=1)
    ax.axhline(BANKROLL, color="red", linestyle="--", linewidth=1)
    ax.set_title(f"Bankroll — Martingale {sim['direction']}")
    ax.set_xlabel("Candle #"); ax.set_ylabel("Bankroll ($)")
    profit_str = f"+${sim['profit']:.2f} ({sim['roi']:+.1f}%)"
    ax.annotate(profit_str, xy=(0.65, 0.08), xycoords="axes fraction",
                fontsize=10, fontweight="bold",
                color="seagreen" if sim["profit"] > 0 else "red")

# Plot 3: Win % at each step
ax3 = fig.add_subplot(gs[0, 2])
for probs, col, lab in [(probs_up, "steelblue", "UP"),
                         (probs_down, "coral", "DOWN")]:
    ax3.plot([p["step"] for p in probs],
             [p["win_pct"] for p in probs],
             marker="o", color=col, label=lab, linewidth=2)
ax3.axhline(50, color="gray", linestyle="--", linewidth=1)
ax3.set_title("Win % at Each Step\n(given all prior steps lost)")
ax3.set_xlabel("Consecutive candle step")
ax3.set_ylabel("Win %"); ax3.legend(fontsize=9)
ax3.set_xticks(range(1, MAX_STEPS + 1))

# Plot 4: Cumulative win %
ax4 = fig.add_subplot(gs[1, 0])
for probs, col, lab in [(probs_up, "steelblue", "UP"),
                         (probs_down, "coral", "DOWN")]:
    ax4.plot([p["step"] for p in probs],
             [p["cumul_pct"] for p in probs],
             marker="o", color=col, label=lab, linewidth=2)
ax4.axhline(80, color="gray", linestyle=":", linewidth=1, label="80%")
ax4.axhline(95, color="gray", linestyle=":", linewidth=1, label="95%")
ax4.set_title("Cumulative % Recovered Within N Steps")
ax4.set_xlabel("Step #"); ax4.set_ylabel("Cumulative %")
ax4.legend(fontsize=8); ax4.set_ylim(0, 105)
ax4.set_xticks(range(1, MAX_STEPS + 1))

# Plot 5: Which step wins
ax5 = fig.add_subplot(gs[1, 1])
for sim, col in [(sim_up, "steelblue"), (sim_down, "coral")]:
    if not sim["which_won"]: continue
    u, c = np.unique(sim["which_won"], return_counts=True)
    offset = -0.2 if sim["direction"] == "UP" else 0.2
    ax5.bar(u + offset, c, width=0.35, color=col, alpha=0.85,
            label=sim["direction"])
ax5.set_title("Which Step the Win Comes On")
ax5.set_xlabel("Step # (0=base $5, 1=$10, 2=$20...)")
ax5.set_ylabel("Count"); ax5.legend(fontsize=9)
ax5.set_xticks(range(MAX_STEPS))

# Plot 6: Exposure ladder
ax6 = fig.add_subplot(gs[1, 2])
steps = list(range(1, MAX_STEPS + 1))
cum   = [BASE_STAKE * (MULTIPLIER**s - 1) / (MULTIPLIER - 1) for s in range(1, MAX_STEPS+1)]
ax6.bar(steps, cum, color=["seagreen" if c <= BANKROLL else "red" for c in cum])
ax6.axhline(BANKROLL, color="red", linestyle="--", linewidth=2,
            label=f"Bankroll ${BANKROLL:.0f}")
ax6.set_title(f"Cumulative Exposure per Step\n(${BASE_STAKE} base, {MULTIPLIER}x)")
ax6.set_xlabel("Step #"); ax6.set_ylabel("Total $ at risk")
ax6.legend(fontsize=9)
for i, (s, v) in enumerate(zip(steps, cum)):
    ax6.text(s, v + 3, f"${v:.0f}", ha="center", fontsize=9)

# Plot 7: No-cap streak distribution
ax7 = fig.add_subplot(gs[2, 0:2])
bins = range(0, max(int(nocap_up.max()), int(nocap_down.max())) + 2)
ax7.hist(nocap_up,   bins=bins, alpha=0.6, color="steelblue", label="UP",
         density=True)
ax7.hist(nocap_down, bins=bins, alpha=0.6, color="coral",     label="DOWN",
         density=True)
ax7.axvline(MAX_STEPS, color="black", linestyle="--", linewidth=2,
            label=f"6-step cap")
ax7.set_title("No-Cap: Distribution of Losses Before First Win\n"
              "(dashed = your 6-step cap — anything right of line = danger zone)")
ax7.set_xlabel("Consecutive losses before win")
ax7.set_ylabel("Density")
ax7.legend(fontsize=9)

# Plot 8: ROI summary
ax8 = fig.add_subplot(gs[2, 2])
labels  = ["Start", f"UP\nMartingale", f"DOWN\nMartingale"]
values  = [BANKROLL, sim_up["final"], sim_down["final"]]
cols    = ["gray", "steelblue", "coral"]
bars    = ax8.bar(labels, values, color=cols, edgecolor="white", width=0.5)
ax8.axhline(BANKROLL, color="red", linestyle="--", linewidth=1.5)
ax8.set_title("Final Bankroll Summary")
ax8.set_ylabel("Bankroll ($)")
for bar, v in zip(bars, values):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f"${v:,.2f}", ha="center", fontsize=10, fontweight="bold")

out = os.path.join(os.path.dirname(__file__), "extended_backtest.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart saved → {out}")
