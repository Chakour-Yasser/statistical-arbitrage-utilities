import sys, warnings, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import config as C
from src.universe import tradable_universe
from src.statarb import run_statarb
from src.backtest import metrics

close = pd.read_parquet(C.DATA_PROC / "spx_prices_adj.parquet")
dv    = pd.read_parquet(C.DATA_PROC / "spx_dollar_volume.parquet")
memb  = pd.read_parquet(C.DATA_PROC / "membership_all.parquet")
memb  = tradable_universe(memb, close)          # identity check: no recycled tickers
common = [c for c in close.columns if c in memb.columns]
close, dv, memb = close[common], dv[common], memb[common]
print(f"universe: {len(common)} tickers after identity validation", flush=True)

out = {}
for tag, null in (("real", False), ("null", True)):
    res = run_statarb(close, dv, memb, null_permute=null, seed=C.SEED)
    m_net, m_gross = metrics(res["net"], res["turnover"]), metrics(res["gross"])
    out[tag] = res
    print(f"\n=== {tag.upper()} ===", flush=True)
    print(f"  days {m_net['days']} | names/day {res.n_names.mean():.0f} | "
          f"positions/day {res.n_pos.mean():.0f} | gross exposure {res.gross_exposure.mean():.2f}")
    print(f"  Sharpe gross {m_gross['sharpe']:+.2f} | net {m_net['sharpe']:+.2f} | "
          f"ann.ret {m_net['ann_return']:+.2%} | maxDD {m_net['max_dd']:.3f} | "
          f"turnover {m_net['turnover']:.1f}/yr", flush=True)

pickle.dump(out, open(C.DATA_PROC / "statarb.pkl", "wb"))
print("\nsaved -> data/processed/statarb.pkl")
