import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import config as C
from src.universe import tradable_universe
from src.statarb import run_statarb
from src.backtest import metrics
close = pd.read_parquet(C.DATA_PROC/"spx_prices_adj.parquet")
dv = pd.read_parquet(C.DATA_PROC/"spx_dollar_volume.parquet")
memb = tradable_universe(pd.read_parquet(C.DATA_PROC/"membership_all.parquet"), close)
common=[c for c in close.columns if c in memb.columns]
close,dv,memb = close[common],dv[common],memb[common]
panel = pd.read_parquet(C.DATA_PROC/"ml_panel.parquet")
def table(f, smooth=1):
    pr=pd.read_parquet(C.DATA_PROC/f)
    sub=panel[panel.date>=pr.date.min()].reset_index(drop=True); assert len(sub)==len(pr)
    t=sub.assign(v=pr["pred"].values).pivot_table(index="date",columns="symbol",values="v")
    return t.ewm(span=smooth).mean() if smooth>1 else t
M=metrics
print(f"{'configuration':<34}{'pos/j':>7}{'turn':>7}{'gross':>7}{'BE bp':>7}{'@1bp':>7}{'@2bp':>7}{'15-20':>7}{'21-26':>7}", flush=True)
def show(lab,r):
    tn=r.turnover.sum()/(len(r)/252); g=M(r["gross"]); n=lambda b: r["gross"]-r["turnover"]*b/1e4
    n2=n(2)
    print(f"{lab:<34}{r.n_pos.mean():>7.0f}{tn:>7.1f}{g['sharpe']:>7.2f}{1e4*g['ann_return']/tn:>7.2f}"
          f"{M(n(1))['sharpe']:>7.2f}{M(n2)['sharpe']:>7.2f}"
          f"{M(n2[n2.index.year<=2020])['sharpe']:>7.2f}{M(n2[n2.index.year>2020])['sharpe']:>7.2f}", flush=True)
B=dict(no_trade_band=1.0, beta_step=1, risk_scale=True)
show("s-score OU + hysteresis (ref)", run_statarb(close,dv,memb,**B))
for fr,sm in ((0.15,1),(0.10,1),(0.15,3)):
    show(f"reseau, quantile {fr:.0%}, EWMA={sm}",
         run_statarb(close,dv,memb,**B, signal_table=table("ml_pred_real.parquet",sm), quantile_frac=fr))
show("reseau null, quantile 15%",
     run_statarb(close,dv,memb,**B, signal_table=table("ml_pred_null.parquet"), quantile_frac=0.15))
