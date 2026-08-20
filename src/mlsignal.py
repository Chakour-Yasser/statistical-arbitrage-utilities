"""
Learned expected residual return, replacing the Ornstein-Uhlenbeck assumption.

Motivation
----------
The OU specification forces E[r | s] to be linear in s. Measured on 415,743
observations it is not: the relation peaks near |s| ~ 1.5 at 3-4.5 bp with
t-statistics around 3, and flattens in the tails. Feeding that misspecified mu
to a convex optimiser makes it size its largest positions where the measured
signal is weakest, which is why the optimiser lost to a crude threshold rule.

The optimiser is not the problem; its input is. This module estimates
E[r_{t+1} | features_t] directly, so the optimiser can be judged on a mu that
matches the data.

Controls
--------
The failure mode of a network on financial cross-sections is overfitting, so
every guard from the rest of the project applies here and one more:

  walk-forward      a model predicting year Y is trained only on data ending
                    before Y begins, and is refitted each year
  scaling           feature means and standard deviations come from the training
                    window alone
  early stopping    on a temporally Later validation slice, never a random split,
                    because a random split leaks the future through cross-
                    sectional correlation on the same date
  small capacity    two hidden layers, weight decay, dropout
  null control      the same pipeline, training included, on a panel whose target
                    is permuted within each date, that destroys the
                    cross-sectional mapping and preserves everything else
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

Features = ["s", "b", "hl", "r1", "r2", "r5", "r10", "r20",
            "vol_ratio", "adv_rank", "drawdown", "disp"]


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden=(64, 32), p_drop: float = 0.2):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_one(xtr, ytr, xva, yva, seed: int, epochs: int = 40,
             lr: float = 1e-3, wd: float = 1e-4, batch: int = 4096) -> MLP:
    torch.manual_seed(seed)
    model = MLP(xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    xtr_t, ytr_t = torch.tensor(xtr), torch.tensor(ytr)
    xva_t, yva_t = torch.tensor(xva), torch.tensor(yva)
    best, best_state, patience = np.inf, None, 0
    n = len(xtr_t)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            loss = lossf(model(xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(xva_t), yva_t).item()
        if v < best - 1e-7:
            best, best_state, patience = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def walk_forward_predict(panel: pd.DataFrame, first_test_year: int = 2018,
                         val_frac: float = 0.2, n_seeds: int = 3,
                         permute_target: bool = False, seed0: int = 0) -> pd.DataFrame:
    """Out-of-sample predictions, one model per test year.

    `permute_target` shuffles y within each date before training, the null.
    It leaves every marginal distribution and the whole time structure intact and
    destroys only the cross-sectional mapping the network is supposed to learn.
    """
    df = panel.copy()
    if permute_target:
        rng = np.random.default_rng(seed0 + 999)
        df["y"] = df.groupby("date")["y"].transform(
            lambda v: v.to_numpy()[rng.permutation(len(v))])

    df["year"] = df["date"].dt.year
    out = []
    for year in range(first_test_year, int(df["year"].max()) + 1):
        tr = df[df["year"] < year]
        te = df[df["year"] == year]
        if len(tr) < 50_000 or te.empty:
            continue
        cut = tr["date"].quantile(1 - val_frac)
        tr_in, tr_va = tr[tr["date"] <= cut], tr[tr["date"] > cut]
        mu_, sd_ = tr_in[Features].mean(), tr_in[Features].std().replace(0, 1)

        def prep(d):
            return ((d[Features] - mu_) / sd_).to_numpy(dtype=np.float32)

        xtr, ytr = prep(tr_in), tr_in["y"].to_numpy(dtype=np.float32)
        xva, yva = prep(tr_va), tr_va["y"].to_numpy(dtype=np.float32)
        xte = prep(te)
        preds = np.zeros(len(te), dtype=np.float64)
        for k in range(n_seeds):                 # seed averaging: cheap variance cut
            m = _fit_one(xtr, ytr, xva, yva, seed=seed0 + 100 * k)
            with torch.no_grad():
                preds += m(torch.tensor(xte)).numpy().astype(np.float64)
        preds /= n_seeds
        res = te[["date", "y", "sigma_step", "s"]].copy()
        res["pred"] = preds
        out.append(res)
        print(f"  {year}: train {len(tr_in):,} val {len(tr_va):,} test {len(te):,} "
              f"| IC {np.corrcoef(preds, res['y'])[0, 1]:+.4f}", flush=True)
    return pd.concat(out, ignore_index=True)
