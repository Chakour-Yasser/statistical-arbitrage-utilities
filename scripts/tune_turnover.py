import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import pandas as pd
from src import config as C
from src.universe import tradable_universe
from src.statarb import run_statarb
from src.backtest import metrics

close = pd.read_parquet(C.DATA_PROC/"spx_prices_adj.parquet")
dv = pd.read_parquet(C.DATA_PROC/"spx_dollar_volume.parquet")
memb = tradable_universe(pd.read_parquet(C.DATA_PROC/"membership_all.parquet"), close)
common = [c for c in close.columns if c in memb.columns]
close, dv, memb = close[common], dv[common], memb[common]
print(f"{'band':>6}{'bstep':>7}{'turn/yr':>10}{'gross':>8}{'@1bp':>8}{'@2bp':>8}{'@3bp':>8}{'@5bp':>8}", flush=True)
for band, bstep in ((0,1),(1.0,1),(2.0,1),(2.0,5),(4.0,5),(6.0,5)):
    r = run_statarb(close, dv, memb, no_trade_band=band, beta_step=bstep)
    tn = r.turnover.sum()/(len(r)/252)
    g = metrics(r["gross"])["sharpe"]
    s = [metrics(r["gross"]-r["turnover"]*b/1e4)["sharpe"] for b in (1,2,3,5)]
    print(f"{band:>6.1f}{bstep:>7d}{tn:>10.1f}{g:>8.2f}" + "".join(f"{x:>8.2f}" for x in s), flush=True)
