"""Shared CFB stack test helpers: fit = 2014-2021 lined games (no 2020), te = 2022-2025 lined games, all features built
by cfb_gap*.py (dumped by cfb_dump.py)."""
import pandas as pd, numpy as np
fit = pd.read_parquet("cfb_fit.parquet"); import os; te = pd.read_parquet(os.environ.get("TE", "cfb_te_all.parquet"))
RC = ['qb_anya', 'qb_lvl', 'qb_new', 'qb_none', 'avail_skill', 'xfer_skill']; D = ['d_' + c for c in RC]
P = ['s1', 's23']; RT = ['d_ro', 'd_rp']
base = ['rm', 'mk'] + D + ['pk', 'kal', 'r0'] + RT + P
v3 = te.model_margin.values; w = te.wk.values <= 3
def run(cols, split=True, y='am', lam=0.0, F=None, Tt=None):
    F = fit if F is None else F; Tt = te if Tt is None else Tt
    pred = np.zeros(len(Tt))
    groups = [(F.wk <= 3, Tt.wk <= 3), (F.wk > 3, Tt.wk > 3)] if split else [(F.wk > 0, Tt.wk > 0)]
    for fm, tm in groups:
        X = np.c_[np.ones(fm.sum()), F.loc[fm, cols].values]; A = X.T @ X + lam * np.diag([0] + [1] * len(cols))
        b = np.linalg.solve(A, X.T @ F[y][fm].values); pred[tm.values] = np.c_[np.ones(tm.sum()), Tt.loc[tm, cols].values] @ b
    return pred
def rep(name, pred, y='am', base=None):
    e = np.abs(pred - te[y].values)
    s = f"{name:44s} all {e.mean():.3f}  wk1-3 {e[w].mean():.3f}  wk4+ {e[~w].mean():.3f}"
    if base is not None:
        d = e - np.abs(base - te[y].values); s += f"  vs v3 {d.mean():+.3f} (t {d.mean() / (d.std() / np.sqrt(len(d))):.1f})"
        s += "  seasons " + " ".join(f"{v:+.2f}" for v in pd.Series(d).groupby(te.season.values).mean())
    print(s)
