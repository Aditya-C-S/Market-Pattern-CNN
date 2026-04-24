# stage2_image_generation.py

import os
import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use("Agg")  # no GUI, just save to disk

# ── Config (must match Stage 1) ───────────────────────────────────────────────
RAW_DIR    = "data/raw"
IMG_DIR    = "data/images"
WINDOW     = 30
LOOKAHEAD  = 5
THRESHOLD  = 0.03
IMG_SIZE   = (128, 128)   # pixels — small enough to train fast, big enough for patterns
TICKERS    = ["SPY", "AAPL", "GLD", "RELIANCE.NS"]

# ── Create output folders ─────────────────────────────────────────────────────
for label in ["UP", "DOWN", "SIDEWAYS"]:
    os.makedirs(os.path.join(IMG_DIR, label), exist_ok=True)

# ── Label logic ───────────────────────────────────────────────────────────────
def get_label(df: pd.DataFrame, start_idx: int) -> str:
    end_idx    = start_idx + WINDOW - 1
    future_idx = end_idx + LOOKAHEAD

    if future_idx >= len(df):
        return None  # not enough future data

    close_now    = df["Close"].iloc[end_idx]
    close_future = df["Close"].iloc[future_idx]
    ret          = (close_future - close_now) / close_now

    if ret > THRESHOLD:
        return "UP"
    elif ret < -THRESHOLD:
        return "DOWN"
    else:
        return "SIDEWAYS"

# ── Chart renderer ────────────────────────────────────────────────────────────
# mplfinance style: no axes, no labels, no titles — pure visual pattern
chart_style = mpf.make_mpf_style(
    marketcolors=mpf.make_marketcolors(
        up="#26a69a",    # teal for bullish
        down="#ef5350",  # red for bearish
        edge="inherit",
        wick="inherit",
        volume="in",
    ),
    gridstyle="",
    facecolor="white",
    figcolor="white",
)

def save_chart(window_df: pd.DataFrame, filepath: str):
    fig, _ = mpf.plot(
        window_df,
        type="candle",
        style=chart_style,
        volume=False,       # no volume bars — keep it pure price pattern
        axisoff=True,       # no axes, ticks, or labels
        returnfig=True,
        figsize=(IMG_SIZE[0] / 100, IMG_SIZE[1] / 100),  # inches @ 100 dpi
    )
    fig.savefig(filepath, dpi=100, bbox_inches="tight", pad_inches=0)
    matplotlib.pyplot.close(fig)

# ── Main loop ─────────────────────────────────────────────────────────────────
total_saved = 0
label_counts = {"UP": 0, "DOWN": 0, "SIDEWAYS": 0}

for ticker in TICKERS:
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    df   = pd.read_csv(path, index_col="Date", parse_dates=True)
    df   = df[["Open", "High", "Low", "Close", "Volume"]]

    print(f"\nProcessing {ticker} ({len(df)} rows)...")
    saved = 0

    for i in range(len(df) - WINDOW - LOOKAHEAD):
        label = get_label(df, i)
        if label is None:
            continue

        window_df = df.iloc[i : i + WINDOW]
        filename  = f"{ticker}_{i:04d}.png"
        filepath  = os.path.join(IMG_DIR, label, filename)

        save_chart(window_df, filepath)
        label_counts[label] += 1
        saved += 1

        if saved % 200 == 0:
            print(f"  {saved} images saved...")

    print(f"  ✓ {ticker} done — {saved} images")
    total_saved += saved

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'─'*40}")
print(f"  Total images saved : {total_saved}")
print(f"  UP                 : {label_counts['UP']}")
print(f"  DOWN               : {label_counts['DOWN']}")
print(f"  SIDEWAYS           : {label_counts['SIDEWAYS']}")
print(f"\n  Saved to: {IMG_DIR}/")
print(f"{'─'*40}")
print("\n✓ Stage 2 complete.")