"""Benter (1994) second-stage blend on our props: y ~ a + b1*logit(market no-vig) + b2*logit(our model).
Fitted on the earlier half of dates, scored on the later half. b2 > 0 with a lower test log loss = our model adds
information the market does not have (Model B core idea). python3 benter_test.py -> results/benter_test.csv"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devig as D
R = os.path.dirname(os.path.abspath(__file__)) + '/../oddspapi/results'
def lg(p): p = np.clip(p, 1e-5, 1 - 1e-5); return np.log(p / (1 - p))
def ll(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def fit(X, y, it=40):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); w += np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(len(w)), X.T @ (y - p))
    return w
rows = []
for sp, f in (('NBA', 'nba_joined.parquet'), ('NHL', 'nhl_joined.parquet'), ('NFL', 'live_nfl_joined.parquet')):
    d = pd.read_parquet(f'{R}/{f}')
    if 'did_not_play' in d: d = d[~d.did_not_play.fillna(False).astype(bool)]
    d = d[d.actual.notna() & (d.actual != d.line) & d.p_model.notna()]
    dc = 'date' if 'date' in d else ('game_date' if 'game_date' in d else 'fid')
    for bk, g in d.groupby('book'):
        for t in ('open', 'close'):
            g = g[(g[f'o_{t}'] > 1) & (g[f'u_{t}'] > 1)]
            if len(g) < 400: continue
            pm = np.array([D.power([a, b])[0] for a, b in zip(g[f'o_{t}'], g[f'u_{t}'])]); y = (g.actual > g.line).astype(float).values
            X = np.c_[lg(pm), lg(g.p_model.values)]
            o = np.argsort(g[dc].astype(str).values, kind='stable'); h = len(o) // 2; tr, te = o[:h], o[h:]
            w = fit(X[tr], y[tr]); pb = 1 / (1 + np.exp(-(w[0] + X[te] @ w[1:])))
            wm = fit(X[tr, :1], y[tr]); pmc = 1 / (1 + np.exp(-(wm[0] + X[te, :1] @ wm[1:])))
            dd = ll(pb, y[te]) - ll(pmc, y[te])
            rows.append(dict(sport=sp, book=bk, time=t, n=len(g), market=ll(pm[te], y[te]).mean(), market_cal=ll(pmc, y[te]).mean(),
                             model=ll(g.p_model.values[te], y[te]).mean(), benter=ll(pb, y[te]).mean(), b_market=w[1], b_model=w[2],
                             benter_vs_marketcal_z=dd.mean() / (dd.std() / np.sqrt(len(dd)))))
T = pd.DataFrame(rows); T.to_csv(os.path.dirname(os.path.abspath(__file__)) + '/results/benter_test.csv', index=False)
pd.set_option('display.width', 200); print(T.round(4).to_string(index=False))
