import sys, warnings, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import config as C
from src.crypto import build_membership
from src.statarb import run_statarb
from src.backtest import metrics
PPY = 365
close = pd.read_parquet(C.DATA_PROC/"crypto_close.parquet")
vol = pd.read_parquet(C.DATA_PROC/"crypto_dollar_volume.parquet")
uni = pd.read_parquet(C.DATA_PROC/"crypto_universe.parquet")
memb = build_membership(uni, close.index)
M = lambda x: metrics(x, periods_per_year=PPY)
CFG = dict(n_factors=15, pca_window=252, no_trade_band=1.0, beta_step=1,
           risk_scale=True, min_adv=1e7, s_open=2.0, periods_per_year=PPY)

real = run_statarb(close, vol, memb, **CFG)
g = M(real["gross"])["sharpe"]
print(f"REAL gross Sharpe {g:+.2f} | {real.n_pos.mean():.0f} positions/day", flush=True)

nulls = []
for sd in (1, 2, 3, 4, 5):
    n = run_statarb(close, vol, memb, null_permute=True, seed=sd, **CFG)
    nulls.append(M(n["gross"])["sharpe"])
    print(f"  null seed {sd}: {nulls[-1]:+.2f}", flush=True)
print(f"NULL mean {np.mean(nulls):+.2f} sd {np.std(nulls):.2f} -> real is "
      f"{(g-np.mean(nulls))/np.std(nulls):.1f} sd above", flush=True)

net = real["gross"] - real["turnover"]*4/1e4
print("\nBy year (net @ 4 bp):", flush=True)
for y, ch in net.groupby(net.index.year):
    if len(ch) < 100: continue
    m = M(ch); print(f"  {y}: Sharpe {m['sharpe']:+6.2f}  ret {m['ann_return']:+8.2%}  DD {m['max_dd']:.3f}", flush=True)
h1, h2 = net[net.index < '2024-01-01'], net[net.index >= '2024-01-01']
print(f"\n  2021-2023: Sharpe {M(h1)['sharpe']:+.2f}\n  2024-2026: Sharpe {M(h2)['sharpe']:+.2f}", flush=True)
pickle.dump({"real": real, "nulls": nulls, "cfg": CFG}, open(C.DATA_PROC/"crypto_validate.pkl","wb"))
