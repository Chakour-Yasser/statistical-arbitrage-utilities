import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src import config as C
from src.universe import tradable_universe
from src.statarb import eigen_portfolios, factor_returns, residual_betas, ou_scores

close = pd.read_parquet(C.DATA_PROC/"spx_prices_adj.parquet")
dv = pd.read_parquet(C.DATA_PROC/"spx_dollar_volume.parquet")
memb = tradable_universe(pd.read_parquet(C.DATA_PROC/"membership_all.parquet"), close)
common=[c for c in close.columns if c in memb.columns]
close,dv,memb = close[common],dv[common],memb[common]
rets = close.pct_change()
dates = close.index
S, R, K = [], [], []
q=None; q_cols=None
for i in range(253, len(dates)-1, 3):          # every 3rd day is plenty for a shape estimate
    t=dates[i]; snaps=memb.index[memb.index<=t]
    if not len(snaps): continue
    win = rets.iloc[i-251:i+1]; adv = dv.iloc[i-251:i+1].median()
    cols=[c for c in memb.columns[memb.loc[snaps[-1]]] if win[c].notna().all() and adv.get(c,0)>=1e7]
    if len(cols)<50: continue
    X = win[cols].values
    if q is None or q_cols!=cols: q,_ = eigen_portfolios(X,15); q_cols=cols
    F = factor_returns(X,q); beta,res = residual_betas(X[-60:],F[-60:]); sc = ou_scores(res)
    # rendement Residuel de la periode suivante, couvert des memes facteurs
    r_next = rets.iloc[i+1][cols].values
    f_next = r_next @ q
    resid_next = r_next - (beta[0] + f_next @ beta[1:])
    ok = np.isfinite(sc["s_score"]) & np.isfinite(resid_next)
    S.append(sc["s_score"][ok]); R.append(resid_next[ok])
    hl = np.log(2)/sc["kappa"]*252; K.append(hl[ok])
S=np.concatenate(S); R=np.concatenate(R); K=np.concatenate(K)
m = np.isfinite(K)&(K>0)&(K<=30)
S,R = S[m], R[m]
print(f"{len(S):,} observations (s, rendement residuel suivant)\n")
edges = np.percentile(S, [0,2,5,10,20,35,50,65,80,90,95,98,100])
print(f"{'bin de s':>16}{'n':>8}{'s moyen':>10}{'E[r|s] bp':>12}{'t-stat':>9}")
for lo,hi in zip(edges[:-1], edges[1:]):
    sel = (S>=lo)&(S<hi)
    if sel.sum()<200: continue
    r = R[sel]; t = r.mean()/(r.std(ddof=1)/np.sqrt(len(r)))
    print(f"[{lo:>6.2f},{hi:>6.2f}]{sel.sum():>8}{S[sel].mean():>10.2f}{1e4*r.mean():>12.2f}{t:>9.2f}")
# linearite : correlation de rang vs correlation lineaire
print(f"\ncorrelation lineaire s vs r  : {np.corrcoef(S,R)[0,1]:+.4f}")
lo,hi = np.percentile(S,[10,90])
mid = (S>lo)&(S<hi)
print(f"  sur le coeur (10-90 pct)   : {np.corrcoef(S[mid],R[mid])[0,1]:+.4f}")
print(f"  sur les queues (<10, >90)  : {np.corrcoef(S[~mid],R[~mid])[0,1]:+.4f}")
