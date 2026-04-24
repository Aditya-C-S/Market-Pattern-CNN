# stage1_data_collection.py

import yfinance as yf
import pandas as pd
import os

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS   = ["SPY", "AAPL", "GLD", "RELIANCE.NS"]   # liquid, lots of history
START     = "2015-01-01"
END       = "2024-01-01"
WINDOW    = 30    # days per chart image (used in Stage 2)
LOOKAHEAD = 5     # days into future for labelling (used in Stage 3)
THRESHOLD = 0.02  # 2% move = Up or Down, else Sideways
OUTPUT_DIR = "data/raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Download ──────────────────────────────────────────────────────────────────
def download_ticker(ticker: str) -> pd.DataFrame:
    print(f"\nDownloading {ticker}...")
    df = yf.download(ticker, start=START, end=END, auto_adjust=True)
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df

# ── Quality check ─────────────────────────────────────────────────────────────
def quality_report(ticker: str, df: pd.DataFrame):
    print(f"\n{'─'*40}")
    print(f"  {ticker}")
    print(f"{'─'*40}")
    print(f"  Rows          : {len(df)}")
    print(f"  Date range    : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    print(f"  Usable windows: {len(df) - WINDOW - LOOKAHEAD}  (≈ chart images)")
    print(f"  Columns       : {list(df.columns)}")
    print(f"  Close min/max : {df['Close'].min():.2f} / {df['Close'].max():.2f}")

# ── Main ──────────────────────────────────────────────────────────────────────
all_data = {}

for ticker in TICKERS:
    df = download_ticker(ticker)
    quality_report(ticker, df)
    
    path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
    df.to_csv(path)
    print(f"  Saved → {path}")
    
    all_data[ticker] = df

print("\n✓ Stage 1 complete. All tickers saved to data/raw/")