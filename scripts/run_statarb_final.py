import sys, warnings, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import config as C
from src.universe import tradable_universe
from src.statarb import run_statarb
from src.backtest import metrics
from src.analysis import subperiod_metrics, rate_exposure

close = pd.read_parquet(C.DATA_PROC/"spx_prices_adj.parquet")
dv = pd.read_parquet(C.DATA_PROC/"spx_dollar_volume.parquet")
memb = tradable_universe(pd.read_parquet(C.DATA_PROC/"membership_all.parquet"), close)
common = [c for c in close.columns if c in memb.columns]
close, dv, memb = close[common], dv[common], memb[common]

CFG = dict(no_trade_band=1.0, beta_step=1)
real = run_statarb(close, dv, memb, **CFG)
null = run_statarb(close, dv, memb, null_permute=True, seed=C.SEED, **CFG)
pickle.dump({"real": real, "null": null, "cfg": CFG}, open(C.DATA_PROC/"statarb_final.pkl","wb"))

tn = real.turnover.sum()/(len(real)/252)
mg = metrics(real["gross"])
print("="*70); print("CROSS-SECTIONAL STAT ARB — S&P 500, 2015-2026".center(70)); print("="*70)
print(f"\nuniverse {real.n_names.mean():.0f} names/day | {real.n_pos.mean():.0f} positions/day "
      f"| gross exposure {real.gross_exposure.mean():.2f} | {len(real)} days")
print(f"\nGROSS  ann.return {mg['ann_return']:+.2%}  vol {mg['ann_vol']:.2%}  "
      f"Sharpe {mg['sharpe']:+.2f}  maxDD {mg['max_dd']:.3f}")
print(f"turnover {tn:.1f}/yr  ->  breakeven cost {1e4*mg['ann_return']/tn:.2f} bp")
print(f"\nNULL CONTROL (common time permutation, factor structure preserved)")
print(f"  gross Sharpe {metrics(null['gross'])['sharpe']:+.2f}   <-- must be ~0")
print("\nNET BY COST ASSUMPTION")
for b in (0.5,1,1.5,2,3,5):
    m = metrics(real["gross"]-real["turnover"]*b/1e4)
    print(f"  {b:4.1f} bp : Sharpe {m['sharpe']:+.2f}  ann {m['ann_return']:+.2%}  DD {m['max_dd']:.3f}")
net2 = real["gross"] - real["turnover"]*2/1e4
print("\nBY YEAR (net @ 2 bp)")
print(subperiod_metrics(net2).round(4).to_string(index=False))
tnx = pd.read_parquet(C.DATA_PROC/"tnx.parquet")["tnx"]
r = rate_exposure(net2, tnx)
print(f"\nrate exposure: beta={r['beta']:+.5f} t={r['t_stat']:+.2f} R2={r['r2']:.4f}")
