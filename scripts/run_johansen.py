import sys, warnings, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import config as C
from src.universe import tradable_universe
from src.johansen import screen_baskets, basket_spread, null_rejection_rate
from src.cointegration import calibrate_zscore, zscore_frozen, half_life
from src.selection import walk_forward_folds, liquidity_filter
from src.pipeline import _universe_at
from src.backtest import pair_positions, metrics

close = pd.read_parquet(C.DATA_PROC / "prices_adj.parquet")
dv    = pd.read_parquet(C.DATA_PROC / "dollar_volume.parquet")
memb  = tradable_universe(pd.read_parquet(C.DATA_PROC / "membership_utilities.parquet"), close)
lp, rets = np.log(close), close.pct_change()
folds = walk_forward_folds(close.index, C.START, C.END)

MAXG, SIZE = 1200, 3
null_rate = null_rejection_rate(n_series=SIZE, n_obs=750, n_sim=1500, seed=7)
print(f"Johansen trace test, {SIZE} series, T=750: null rejection rate = {100*null_rate:.2f} % "
      f"(nominal 5 %)\n", flush=True)

rows, curves = [], []
for f in folds:
    tk = liquidity_filter(dv, f["is"], _universe_at(memb, f["selection_date"]))
    sc = screen_baskets(lp.loc[f["is"]], tk, size=SIZE, max_groups=MAXG)
    if sc.empty:
        continue
    n_tested = sc.attrs["n_groups_tested"]
    found = int((sc["rank"] >= 1).sum())
    expected = null_rate * n_tested

    # keep the strongest, with the same tradability discipline as the pairs book
    cand = sc[sc["rank"] >= 1].sort_values("excess", ascending=False).head(20)
    legs = []
    for _, r in cand.iterrows():
        w = pd.Series(r["weights"])
        s_is = basket_spread(lp.loc[f["is"]], w)
        hl = half_life(s_is.values)
        if not (2.0 <= (hl if np.isfinite(hl) else 1e9) <= 30.0):
            continue
        mu, sd = calibrate_zscore(s_is)
        z = zscore_frozen(basket_spread(lp.loc[f["oos"]], w), mu, sd)
        pos = pair_positions(z)
        basket_ret = (rets[w.index].loc[f["oos"]] * w).sum(axis=1)
        held = pos.shift(1).fillna(0.0)
        turn = (pos - pos.shift(1).fillna(0.0)).abs() * np.abs(w).sum()
        net = held * basket_ret - turn * 5 / 1e4
        legs.append(pd.DataFrame({"net": net, "turnover": turn}))

    m = metrics(sum(legs)["net"] / len(legs), sum(legs)["turnover"] / len(legs)) if legs else {}
    rows.append({"fold": f["fold"], "groups_total": sc.attrs["n_groups_total"],
                 "groups_tested": n_tested, "rank>=1": found,
                 "expected_null": round(expected), "excess": round(found - expected),
                 "traded": len(legs), "sharpe": m.get("sharpe", np.nan),
                 "ann_return": m.get("ann_return", np.nan), "max_dd": m.get("max_dd", np.nan)})
    if legs:
        curves.append(sum(legs)["net"] / len(legs))
    print(rows[-1], flush=True)

tbl = pd.DataFrame(rows)
print("\n" + tbl.to_string(index=False))
if curves:
    stitched = pd.concat(curves).sort_index()
    print("\nSTITCHED baskets:", {k: round(v, 4) if isinstance(v, float) else v
                                  for k, v in metrics(stitched).items()})
    pickle.dump({"table": tbl, "curve": stitched, "null_rate": null_rate},
                open(C.DATA_PROC / "johansen.pkl", "wb"))
