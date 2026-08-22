"""Dump a causal (features, next residual return) panel for supervised learning."""
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
rets = close.pct_change(); dates = close.index
rows=[]; q=None; q_cols=None
HORIZONS = [1, 2, 3, 5, 10]      # cibles cumulees, une seule passe sur les donnees
HMAX = max(HORIZONS)
for i in range(253, len(dates)-HMAX-1):
    t=dates[i]; snaps=memb.index[memb.index<=t]
    if not len(snaps): continue
    win=rets.iloc[i-251:i+1]; adv=dv.iloc[i-251:i+1].median()
    cols=[c for c in memb.columns[memb.loc[snaps[-1]]] if win[c].notna().all() and adv.get(c,0)>=1e7]
    if len(cols)<50: continue
    X=win[cols].values
    if q is None or q_cols!=cols or (i%5==0): q,_=eigen_portfolios(X,15); q_cols=cols
    F=factor_returns(X,q); beta,res=residual_betas(X[-60:],F[-60:]); sc=ou_scores(res)
    cum=np.cumsum(res,axis=0)
    hl=np.log(2)/sc["kappa"]*252
    sd=sc["sigma_step"]
    # Cibles : rendement residuel Cumule sur k jours, couverture figee a la date i.
    #
    # Un seul parcours produit tous les horizons, ce qui donne la courbe de
    # decroissance de l'information plutot qu'un reglage a l'aveugle. Deux
    # corrections y sont incorporees :
    #   alpha + epsilon  une position couverte encaisse l'intercept, qui n'est
    #                    pas couvrable ; l'exclure surestime l'encaissable, et le
    #                    biais est systematique puisqu'une strategie de reversion
    #                    achete les noms dont l'alpha recent est negatif.
    #   horizon k        une prevision a un jour se perime en un jour et impose
    #                    de trader chaque jour ; un horizon plus long achete de
    #                    la persistance en payant du pouvoir predictif.
    cumres=np.zeros(len(cols)); targets={}
    for k in range(1, HMAX+1):
        r_k=rets.iloc[i+k][cols].values
        f_k=np.nan_to_num(r_k,nan=0.0)@q
        cumres=cumres+(r_k-f_k@beta[1:])
        if k in HORIZONS: targets[k]=cumres.copy()
    with np.errstate(invalid="ignore",divide="ignore"):
        ys={f"y{k}": targets[k]/np.where(sd>0, sd*np.sqrt(k), np.nan) for k in HORIZONS}
    y=ys["y1"]
    ok=np.all([np.isfinite(ys[f"y{k}"]) for k in HORIZONS],axis=0)&np.isfinite(sc["s_score"])&np.isfinite(hl)&(hl>0)&(hl<=30)
    if ok.sum()<20: continue
    df=pd.DataFrame({k:v[ok] for k,v in feat.items()})
    df["symbol"]=np.array(cols)[ok]
    for k in HORIZONS: df[f"y{k}"]=ys[f"y{k}"][ok]
    df["y"]=ys["y1"][ok]; df["date"]=t; df["sigma_step"]=sd[ok]
    df["disp"]=np.nanstd(sc["s_score"][ok])
    rows.append(df)
    if i%400==0: print(f"  {t:%Y-%m} rows={sum(len(x) for x in rows):,}", flush=True)
panel=pd.concat(rows, ignore_index=True).dropna()
panel.to_parquet(C.DATA_PROC/"ml_panel.parquet")
print(f"\npanel: {len(panel):,} rows x {panel.shape[1]} cols, "
      f"{panel.date.min():%Y-%m} -> {panel.date.max():%Y-%m}")
