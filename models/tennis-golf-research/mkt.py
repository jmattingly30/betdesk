"""Market rating: each player's strength learned from PAST Pinnacle closing prices (never the current match's line).
Kalman on the logit scale: close logit(A beats B) ~ m_A - m_B. Unseen players fall back to our model.
Then: does market rating + our model predict (1) the close better than the open, (2) results better than the close?"""
import json, numpy as np, pandas as pd
from collections import defaultdict
from bridge import norm


def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


D = pd.read_pickle('../../op/analysis_rows.pkl')[['fid', 'start', 'level', 'win', 'fair_open', 'fair_close']]
P = pd.concat([pd.read_parquet(f'../data/pred26_{t}.parquet') for t in ('atp', 'wta')])[['fid', 's1']].rename(columns={'s1': 'p_model'})
F = pd.concat([pd.read_parquet(f'../data/feat26_{t}.parquet') for t in ('atp', 'wta')])[['fid', 'a', 'b']]
# all 2026 matches (with or without prices) in time order; market updates only where a close exists
A = pd.concat([pd.read_parquet(f'../data/feat26_{t}.parquet') for t in ('atp', 'wta')])[['fid', 'start', 'a', 'b', 'cat']]
A = A.merge(D[['fid', 'fair_open', 'fair_close', 'win']], on='fid', how='left').merge(P, on='fid', how='left').sort_values('start')
m = defaultdict(float); v = defaultdict(lambda: 1.0); last = {}
Q, R_OBS, DRIFT = 0.0, 0.05, 0.002
rows = []
for r in A.itertuples(index=False):
    a, b = r.a, r.b
    for p in (a, b):
        if p in last:
            v[p] = min(1.0, v[p] + DRIFT * max((r.start - last[p]).days, 0))
    seen = (a in last) and (b in last)
    mk = m[a] - m[b]
    rows.append(dict(fid=r.fid, start=r.start, cat=r.cat, mk=mk, va=v[a], vb=v[b], seen=seen))
    if r.fair_close == r.fair_close:
        # prior for unseen: anchor on our model so the rating starts from it
        pm = r.p_model if r.p_model == r.p_model else 0.5
        for p, sgn in ((a, 1), (b, -1)):
            if p not in last:
                m[p] = 0.0; v[p] = 1.0
        prior = m[a] - m[b]
        if not seen:
            prior = lg(pm); m[a] = prior / 2 if a not in last else m[a]; m[b] = -prior / 2 if b not in last else m[b]
        obs = lg(r.fair_close)
        resid = obs - (m[a] - m[b])
        S = v[a] + v[b] + R_OBS
        m[a] += v[a] / S * resid; m[b] -= v[b] / S * resid
        v[a] *= 1 - v[a] / S; v[b] *= 1 - v[b] / S
        last[a] = last[b] = r.start
M = pd.DataFrame(rows).merge(A[['fid', 'fair_open', 'fair_close', 'win', 'p_model']], on='fid')
z = M[M.fair_close.notna() & M.seen & (M.va < 0.3) & (M.vb < 0.3)]
print('priced matches', M.fair_close.notna().sum(), '| both players already market-rated', len(z))
if len(z) > 30:
    pm = 1 / (1 + np.exp(-z.mk))
    print('predict the close (MAE in prob): open', round((z.fair_open - z.fair_close).abs().mean(), 4), '| market rating', round((pm - z.fair_close).abs().mean(), 4),
          '| our model', round((z.p_model - z.fair_close).abs().mean(), 4))
    w = z.win.notna()
    print('results log loss: close', round(ll(z.fair_close[w], z.win[w]), 4), 'open', round(ll(z.fair_open[w], z.win[w]), 4), 'market rating', round(ll(pm[w], z.win[w]), 4), 'model', round(ll(z.p_model[w], z.win[w]), 4))
M.to_parquet('../data/mkt26.parquet')
