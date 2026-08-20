import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from src import config as C
from src.mlsignal import walk_forward_predict
torch.set_num_threads(4)

panel = pd.read_parquet(C.DATA_PROC / "ml_panel.parquet")
print(f"panel {len(panel):,} rows\n", flush=True)

print("=== real ===", flush=True)
real = walk_forward_predict(panel, first_test_year=2018, n_seeds=3, seed0=C.Seed % 1000)
real.to_parquet(C.DATA_PROC / "ml_pred_real.parquet")

print("\n=== null (target permuted within each date) ===", flush=True)
null = walk_forward_predict(panel, first_test_year=2018, n_seeds=3,
                            permute_target=True, seed0=C.Seed % 1000)
null.to_parquet(C.DATA_PROC / "ml_pred_null.parquet")

def ic(d, lab):
    g = d.groupby("date").apply(lambda x: np.corrcoef(x["pred"], x["y"])[0,1]
                                if len(x) > 5 else np.nan).dropna()
    t = g.mean()/(g.std(ddof=1)/np.sqrt(len(g)))
    print(f"{lab:6s} IC quotidien moyen {g.mean():+.4f}  ecart-type {g.std():.4f}  "
          f"t={t:+.2f}  n={len(g)}", flush=True)
    return g
print()
ic(real, "real"); ic(null, "null")
