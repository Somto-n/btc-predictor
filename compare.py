"""
Main runner: fetches data, builds features, runs all models through
walk-forward backtest, prints comparison table, and saves charts.

Run:  cd btc_predictor && python compare.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(__file__))

from config   import TRAIN_WINDOW, TEST_WINDOW, STEP, SEQUENCE_LEN
from data_feed import fetch_coinbase_candles
from features  import build_features, FEATURE_COLS
from models    import MODEL_REGISTRY
from backtest  import walk_forward, BacktestResult


# ─── 1. Data ──────────────────────────────────────────────────────────────────
print("=" * 65)
print("BTC 5-MIN BINARY DIRECTION PREDICTOR")
print("=" * 65)

raw  = fetch_coinbase_candles()
data = build_features(raw)
from features import FEATURE_COLS   # re-import after build_features sets it
print(f"\nDataset: {len(data)} candles  |  {len(FEATURE_COLS)} features")
print(f"Target balance: UP={data['target'].mean()*100:.1f}%  "
      f"DOWN={(1-data['target'].mean())*100:.1f}%\n")


# ─── 2. Run all models ────────────────────────────────────────────────────────
results: list[BacktestResult] = []

for spec in MODEL_REGISTRY:
    print(f"Running {spec['name']}...")
    seq = SEQUENCE_LEN if spec["is_lstm"] else 0
    r = walk_forward(
        df           = data,
        feature_cols = FEATURE_COLS,
        train_window = TRAIN_WINDOW,
        test_window  = TEST_WINDOW,
        step         = STEP,
        build_model_fn = spec["fn"],
        model_name   = spec["name"],
        sequence_len = seq,
    )
    results.append(r)


# ─── 3. Console summary ───────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("RESULTS SUMMARY")
print(f"{'='*90}")
hdr = (f"{'Model':<22} {'Acc%':>6} {'UP acc%':>8} {'DN acc%':>8} "
       f"{'MaxLoss ALL':>12} {'MaxLoss UP':>11} {'MaxLoss DN':>11} {'AvgStreak':>10}")
print(hdr)
print("-" * 90)
for r in results:
    print(
        f"{r.name:<22} "
        f"{r.accuracy*100:>5.1f}% "
        f"{r.up_accuracy*100:>7.1f}% "
        f"{r.down_accuracy*100:>7.1f}% "
        f"{r.max_consec_loss_all:>12} "
        f"{r.max_consec_loss_up:>11} "
        f"{r.max_consec_loss_down:>11} "
        f"{r.avg_consec_loss:>9.1f}"
    )

print(f"\n{'='*90}")
print("LOSING STREAK PERCENTILES (consecutive wrong predictions)")
print(f"{'='*90}")
hdr2 = f"{'Model':<22} {'p50':>5} {'p75':>5} {'p90':>5} {'p95':>5} {'p99':>5} {'MAX':>5}"
print(hdr2)
print("-" * 55)
for r in results:
    p = r.streak_pcts
    print(f"{r.name:<22} {p[50]:>5} {p[75]:>5} {p[90]:>5} {p[95]:>5} {p[99]:>5} {r.max_consec_loss_all:>5}")

print(f"\n{'='*90}")
print("UP vs DOWN BREAKDOWN")
print(f"{'='*90}")
for r in results:
    print(f"  {r.name}:")
    print(f"    UP   predictions → accuracy {r.up_accuracy*100:.1f}%  |  max consec losses {r.max_consec_loss_up}")
    print(f"    DOWN predictions → accuracy {r.down_accuracy*100:.1f}%  |  max consec losses {r.max_consec_loss_down}")


# ─── 4. Charts ────────────────────────────────────────────────────────────────
names  = [r.name for r in results]
colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

fig = plt.figure(figsize=(20, 14))
fig.suptitle("BTC 5-Min Binary Direction Predictor — Walk-Forward Backtest Results",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

# Plot 1: Overall accuracy
ax1 = fig.add_subplot(gs[0, 0])
accs = [r.accuracy * 100 for r in results]
bars = ax1.bar(names, accs, color=colors)
ax1.axhline(50, color="red", linestyle="--", linewidth=1, label="Random (50%)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Overall Accuracy")
ax1.tick_params(axis="x", rotation=25)
ax1.set_ylim(45, max(accs) + 5)
ax1.legend(fontsize=8)
for bar, v in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f"{v:.1f}%", ha="center", fontsize=8)

# Plot 2: UP vs DOWN accuracy side by side
ax2 = fig.add_subplot(gs[0, 1])
x   = np.arange(len(names))
w   = 0.35
up_accs = [r.up_accuracy * 100   for r in results]
dn_accs = [r.down_accuracy * 100 for r in results]
ax2.bar(x - w/2, up_accs, w, label="UP pred",   color="steelblue")
ax2.bar(x + w/2, dn_accs, w, label="DOWN pred", color="coral")
ax2.axhline(50, color="red", linestyle="--", linewidth=1)
ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=25)
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("UP vs DOWN Prediction Accuracy")
ax2.legend(fontsize=8)

# Plot 3: Max consecutive losses — ALL vs UP vs DOWN
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(len(names))
w = 0.25
ax3.bar(x - w,   [r.max_consec_loss_all  for r in results], w, label="All",  color="dimgray")
ax3.bar(x,       [r.max_consec_loss_up   for r in results], w, label="UP",   color="steelblue")
ax3.bar(x + w,   [r.max_consec_loss_down for r in results], w, label="DOWN", color="coral")
ax3.set_xticks(x); ax3.set_xticklabels(names, rotation=25)
ax3.set_ylabel("Max Consecutive Losses")
ax3.set_title("Max Losing Streak (All / UP / DOWN)")
ax3.legend(fontsize=8)

# Plot 4: Losing streak distribution box plot
ax4 = fig.add_subplot(gs[1, 0:2])
streak_data = []
for r in results:
    wrong  = [1 if p != a else 0 for p, a in zip(r.all_preds, r.all_actuals)]
    runs, cur = [], 0
    for w in wrong:
        if w: cur += 1
        else:
            if cur: runs.append(cur)
            cur = 0
    if cur: runs.append(cur)
    streak_data.append(runs if runs else [0])

bp = ax4.boxplot(streak_data, tick_labels=names, patch_artist=True, notch=False)
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax4.set_ylabel("Losing Streak Length")
ax4.set_title("Losing Streak Distribution (box=25–75th pct, whiskers=5–95th pct)")
ax4.tick_params(axis="x", rotation=20)

# Plot 5: Cumulative correct predictions over time (equity-style)
ax5 = fig.add_subplot(gs[1, 2])
for r, color in zip(results, colors):
    correct = [1 if p == a else -1 for p, a in zip(r.all_preds, r.all_actuals)]
    cumsum  = np.cumsum(correct)
    ax5.plot(cumsum, label=r.name, color=color, linewidth=1.5)
ax5.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax5.set_xlabel("Prediction #")
ax5.set_ylabel("Cumulative (correct - wrong)")
ax5.set_title("Prediction Equity Curve")
ax5.legend(fontsize=7, loc="upper left")

out_path = os.path.join(os.path.dirname(__file__), "btc_predictor_results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nChart saved → {out_path}")
