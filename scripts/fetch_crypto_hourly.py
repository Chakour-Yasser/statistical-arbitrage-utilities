import sys, warnings, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd
from src import config as C
from src.crypto import fetch_klines

Start, End, Top = "2022-01-01", "2026-06-30", 150
dv = pd.read_parquet(C.DATA_PROC / "crypto_dollar_volume.parquet")
uni = pd.read_parquet(C.DATA_PROC / "crypto_universe.parquet")
# rank by median dollar volume over the daily sample: liquidity is the binding
# constraint at hourly horizons, and the tercile test showed illiquid names carry
# no signal in either market
med = dv.loc[dv.index >= Start].median().dropna().sort_values(ascending=False)
syms = [s for s in med.index[:Top]]
print(f"fetching 1h bars for {len(syms)} most liquid perpetuals, {Start} -> {End}", flush=True)

closes, vols, missing = {}, {}, []
for i, s in enumerate(syms, 1):
    df = fetch_klines(s, Start, End, interval="1h")
    if df is None or df.empty:
        missing.append(s); continue
    closes[s], vols[s] = df["close"], df["quote_volume"]
    if i % 20 == 0:
        print(f"  {i}/{len(syms)}  rows so far={len(closes[s])}", flush=True)
    time.sleep(0.05)

close = pd.DataFrame(closes).sort_index()
vol = pd.DataFrame(vols).sort_index().reindex(columns=close.columns)
close.to_parquet(C.DATA_PROC / "crypto_close_1h.parquet")
vol.to_parquet(C.DATA_PROC / "crypto_dollar_volume_1h.parquet")
print(f"\nhourly: {close.shape[0]} bars x {close.shape[1]} symbols "
      f"({close.index[0]} -> {close.index[-1]})")
print(f"missing: {len(missing)}", flush=True)
