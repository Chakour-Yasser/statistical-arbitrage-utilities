import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd, yfinance as yf
from src import config as C

memb = pd.read_parquet(C.DATA_PROC / "membership_all.parquet")
tickers = sorted(memb.columns[memb.any()])
print(f"downloading {len(tickers)} tickers", flush=True)
raw = yf.download(tickers, start=C.Start, end=C.End, auto_adjust=True,
                  progress=False, threads=True, actions=False)
close, vol = raw["Close"], raw["Volume"]
close = close.dropna(axis=1, how="all")
vol = vol.reindex(columns=close.columns)
(close * vol).to_parquet(C.DATA_PROC / "spx_dollar_volume.parquet")
close.to_parquet(C.DATA_PROC / "spx_prices_adj.parquet")
print(f"prices: {close.shape[0]} sessions x {close.shape[1]} tickers", flush=True)
print(f"missing from vendor: {len(tickers) - close.shape[1]}", flush=True)
