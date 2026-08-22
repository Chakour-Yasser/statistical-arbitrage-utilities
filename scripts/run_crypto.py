import sys, warnings, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import config as C
from src.crypto import build_membership, validate_listing_dates
from src.statarb import run_statarb
from src.backtest import metrics
from src.analysis import subperiod_metrics

PPY = 365          # perpetuals trade every day
close = pd.read_parquet(C.DATA_PROC / "crypto_close.parquet")
vol   = pd.read_parquet(C.DATA_PROC / "crypto_dollar_volume.parquet")
uni   = pd.read_parquet(C.DATA_PROC / "crypto_universe.parquet")

val = validate_listing_dates(close, uni)
bad = val[val.status_check == "PRE_ONBOARD_DATA"]
print(f"listing-date validation: {len(val)} symbols, {len(bad)} with data before onboarding")
keep = [s for s in close.columns if s not in set(bad["symbol"])]
close, vol = close[keep], vol[keep]
memb = build_membership(uni[uni.symbol.isin(keep)], close.index)
print(f"universe: {close.shape[1]} symbols, {close.shape[0]} days, "
      f"{memb.sum(axis=1).min()}-{memb.sum(axis=1).max()} live per day\n", flush=True)

CFG = dict(no_trade_band=1.0, beta_step=1, risk_scale=True, n_factors=15,
           min_adv=1e7, periods_per_year=PPY)
real = run_statarb(close, vol, memb, **CFG)
null = run_statarb(close, vol, memb, null_permute=True, seed=C.SEED, **CFG)
pickle.dump({"real": real, "null": null, "cfg": CFG}, open(C.DATA_PROC/"crypto_statarb.pkl","wb"))

M = lambda x, t=None: metrics(x, t, periods_per_year=PPY)
tn = real.turnover.sum() / (len(real) / PPY)
mg = M(real["gross"])
print("="*68); print("CRYPTO Perpetuals, Cross-Sectional Stat Arb".center(68)); print("="*68)
print(f"\n{real.n_names.mean():.0f} symbols/day | {real.n_pos.mean():.0f} positions/day | "
      f"{len(real)} days ({real.index[0].date()} -> {real.index[-1].date()})")
print(f"\nGROSS  ann {mg['ann_return']:+.2%}  vol {mg['ann_vol']:.2%}  "
      f"Sharpe {mg['sharpe']:+.2f}  maxDD {mg['max_dd']:.3f}")
print(f"turnover {tn:.1f}/yr  ->  breakeven cost {1e4*mg['ann_return']/tn:.2f} bp")
print(f"\nnull Control  gross Sharpe {M(null['gross'])['sharpe']:+.2f}   <-- must be ~0")
print("\nNET BY Cost")
for b in (2, 5, 10, 15, 20):
    m = M(real["gross"] - real["turnover"]*b/1e4)
    print(f"  {b:5.1f} bp : Sharpe {m['sharpe']:+.2f}  ann {m['ann_return']:+.2%}  DD {m['max_dd']:.3f}")
net = real["gross"] - real["turnover"]*10/1e4
print("\nBY Year (net @ 10 bp)")
r = subperiod_metrics(net)
r["sharpe"] = r["sharpe"] * np.sqrt(PPY/252); r["ann_return"] = r["ann_return"] * PPY/252
print(r.round(4).to_string(index=False))
