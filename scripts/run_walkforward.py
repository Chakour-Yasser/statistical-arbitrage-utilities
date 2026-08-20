import sys, warnings, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd
from src import config as C
from src.pipeline import run_walk_forward

close = pd.read_parquet(C.DATA_PROC / "prices_adj.parquet")
dv    = pd.read_parquet(C.DATA_PROC / "dollar_volume.parquet")
memb  = pd.read_parquet(C.DATA_PROC / "membership_utilities.parquet")
from src.universe import tradable_universe
memb = tradable_universe(memb, close)

out = {}
for tag, use_regime in (("no_regime", False), ("regime", True)):
    print(f"=== {tag} ===", flush=True)
    tbl, curves = run_walk_forward(close, dv, memb, use_regime=use_regime)
    out[tag] = {"table": tbl, "curves": curves}
    print(tbl.to_string(index=False), flush=True)

with open(C.DATA_PROC / "walkforward.pkl", "wb") as f:
    pickle.dump(out, f)
print("\nsaved -> data/processed/walkforward.pkl")
