import sys, warnings, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd
from src import config as C
from src.crypto import perpetual_universe, fetch_klines

START, END = "2019-09-01", "2026-06-30"
uni = perpetual_universe()
uni.to_parquet(C.DATA_PROC / "crypto_universe.parquet")
print(f"{len(uni)} perpetuals ({(uni.status=='TRADING').sum()} live, "
      f"{(uni.status=='SETTLING').sum()} settling)", flush=True)

closes, vols, missing = {}, {}, []
for i, s in enumerate(uni["symbol"], 1):
    df = fetch_klines(s, START, END)
    if df is None or df.empty:
        missing.append(s); continue
    closes[s], vols[s] = df["close"], df["quote_volume"]
    if i % 50 == 0:
        print(f"  {i}/{len(uni)}  ok={len(closes)} missing={len(missing)}", flush=True)
    time.sleep(0.05)

close = pd.DataFrame(closes).sort_index()
vol = pd.DataFrame(vols).sort_index().reindex(columns=close.columns)
close.to_parquet(C.DATA_PROC / "crypto_close.parquet")
vol.to_parquet(C.DATA_PROC / "crypto_dollar_volume.parquet")
print(f"\nprices: {close.shape[0]} days x {close.shape[1]} symbols "
      f"({close.index[0].date()} -> {close.index[-1].date()})")
print(f"purged from the API (irrecoverable): {len(missing)}", flush=True)
