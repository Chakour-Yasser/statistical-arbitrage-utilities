import sys, warnings, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import config as C
from src.analysis import (cost_sensitivity, drawdown_attribution, rate_exposure,
                          subperiod_metrics, summary_table)
from src.backtest import metrics

d = pickle.load(open(C.DATA_PROC / "walkforward.pkl", "rb"))
tnx = pd.read_parquet(C.DATA_PROC / "tnx.parquet")["tnx"]
W = 78

def head(t): print("\n" + "=" * W); print(t.center(W)); print("=" * W)

head("1. NAIVE vs CORRECTED SELECTION  (stitched OOS, 10 folds)")
for tag in ("no_regime", "regime"):
    print(f"\n-- {tag} --")
    tbl = summary_table({m: b["net"] for m, b in d[tag]["curves"].items()})
    print(tbl.round(4).to_string(index=False))

head("2. WHAT COSTS TAKE  (naive selection, no regime)")
bk = d["no_regime"]["curves"]["naive"]
cs = cost_sensitivity(bk["gross"], bk["turnover"])
print(cs.round(4).to_string(index=False))
z = cs[cs.sharpe > 0]
print(f"\nbreakeven cost: the net Sharpe is already negative at 0 bp"
      if z.empty else f"\nbreakeven cost ~ {z.cost_bps.max()} bp")
print(f"gross Sharpe {metrics(bk['gross'])['sharpe']:+.3f}  ->  net at 5 bp "
      f"{metrics(bk['net'])['sharpe']:+.3f}   (cost of {metrics(bk['gross'])['sharpe']-metrics(bk['net'])['sharpe']:.3f})")

head("3. SIGNAL DECAY  (naive, no regime, by calendar year)")
print(subperiod_metrics(bk["net"]).round(4).to_string(index=False))

head("4. WHERE THE DRAWDOWN COMES FROM  (regime flags, naive selection)")
bkr = d["regime"]["curves"]["naive"]
flagged = bkr["broken_share"] > 0.5
att = drawdown_attribution(bkr["net"], flagged)
for k, v in att.items():
    print(f"  {k:32s} {v: .4f}")
print("\n  -- same book WITHOUT the regime exit, flags used only as a label --")
common = bk.index.intersection(bkr.index)
att2 = drawdown_attribution(bk["net"].loc[common], flagged.loc[common])
for k, v in att2.items():
    print(f"  {k:32s} {v: .4f}")

head("5. IS IT A DISGUISED DURATION BET?  (P&L vs 10y yield changes)")
for tag in ("no_regime", "regime"):
    for m in ("naive", "bh"):
        if m not in d[tag]["curves"]:
            continue
        r = rate_exposure(d[tag]["curves"][m]["net"], tnx)
        print(f"  {tag:10s} {m:6s}  beta={r['beta']:+.5f}  t={r['t_stat']:+.2f}  "
              f"R2={r['r2']:.4f}  n={r['n']}")
print("\n  |t| < 2 means the P&L is not explained by rate moves.")

head("6. REGIME EXIT: EFFECT SUMMARY")
a = d["no_regime"]["table"].set_index("fold"); b = d["regime"]["table"].set_index("fold")
for m in ("naive", "bh"):
    ca, cb = d["no_regime"]["curves"].get(m), d["regime"]["curves"].get(m)
    if ca is None or cb is None: continue
    ma, mb = metrics(ca["net"], ca["turnover"]), metrics(cb["net"], cb["turnover"])
    print(f"\n  {m}:")
    print(f"    {'':16s}{'no regime':>12}{'regime':>12}{'change':>12}")
    for k, fmt in (("sharpe","{:+.3f}"),("ann_return","{:+.2%}"),("max_dd","{:+.4f}"),("turnover","{:.2f}")):
        print(f"    {k:16s}{fmt.format(ma[k]):>12}{fmt.format(mb[k]):>12}"
              f"{fmt.format(mb[k]-ma[k]):>12}")
