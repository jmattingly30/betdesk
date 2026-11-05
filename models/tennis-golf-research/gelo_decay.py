"""Tune the games-won rating: update size, cap, surface version, and set-share vs game-share. Graded 2018-24 with
elo + surface elo + variant (logistic refit yearly on prior 8 years)."""
import numpy as np, pandas as pd, sys
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
tour = sys.argv[1]
F = pd.read_parquet(f'../data/feat_{tour}.parquet', columns=['date', 'level', 'surface', 'a', 'b', 'y', 'ret', 'elo_a', 'elo_b', 'elos_a', 'elos_b', 'n_a', 'n_b', 'games_w', 'games_l'])
A, B, Y, R, GW, GL, S = F.a.values, F.b.values, F.y.values, F.ret.values, F.games_w.values, F.games_l.values, F.surface.values
D = (F.date.values.astype('datetime64[D]').astype(np.int64))
def run(scale, cap, off, surf=False, hl=None, mode='count'):
    g = defaultdict(float); ng = defaultdict(float); out = np.zeros(len(F)); last = {}
    for i in range(len(F)):
        if hl:
            for p in (A[i], B[i]):
                if p in last:
                    dt = D[i] - last[p]
                    if mode == 'count': ng[p] *= 0.5 ** (dt / hl)
                    else: g[p] *= 0.5 ** (dt / hl)
                last[p] = D[i]
        a, b = (A[i], S[i]) if surf else A[i], (B[i], S[i]) if surf else B[i]
        out[i] = g[a] - g[b]
        tot = GW[i] + GL[i]
        if tot == tot and tot > 5 and not R[i]:
            sa = (GW[i] if Y[i] == 1 else GL[i]) / tot
            e = 1 / (1 + np.exp(-(g[a] - g[b])))
            ka = scale / (min(ng[a], cap) + off) ** 0.5; kb = scale / (min(ng[b], cap) + off) ** 0.5
            g[a] += ka * (sa - e); g[b] -= kb * (sa - e); ng[a] += 1; ng[b] += 1
    return out
F['elo'] = lg(1 / (1 + 10 ** ((F.elo_b - F.elo_a) / 400))); F['elos'] = lg(1 / (1 + 10 ** ((F.elos_b - F.elos_a) / 400)))
lvmap = {'Grand Slams': 'Slam', 'Masters': 'Tour', '250 or 500': 'Tour', 'Challenger': 'Challenger', 'Futures': 'ITF', 'Premier': 'Tour', 'Premier Mandatory': 'Tour', 'International': 'Tour', 'C10': 'ITF', 'C25': 'ITF', 'C50': 'ITF', 'C75': 'ITF', 'C100': 'ITF'}
F['lv'] = F.level.map(lvmap); F['yr'] = F.date.dt.year
use = (F.n_a >= 5) & (F.n_b >= 5) & ~F.ret
VAR = {'base 3.6/60/8': dict(scale=3.6, cap=60, off=8), 'count hl 180d': dict(scale=3.6, cap=60, off=8, hl=180), 'count hl 365d': dict(scale=3.6, cap=60, off=8, hl=365),
       'count hl 730d': dict(scale=3.6, cap=60, off=8, hl=730), 'rating hl 730d': dict(scale=3.6, cap=60, off=8, hl=730, mode='rating')}
_OLD = {'base 3.6/60/8': dict(scale=3.6, cap=60, off=8), 'scale 2.4': dict(scale=2.4, cap=60, off=8), 'scale 5': dict(scale=5, cap=60, off=8),
       'cap 150': dict(scale=3.6, cap=150, off=8), 'cap 30': dict(scale=3.6, cap=30, off=8)}
cols = {}
for k, v in VAR.items(): F['g_' + k] = run(**v); cols[k] = ['elo', 'elos', 'g_' + k]

res = {}
G = F[use]
for k, c in cols.items():
    p = pd.Series(np.nan, G.index)
    for yr in range(2018, 2025):
        tr = (G.yr < yr) & (G.yr >= yr - 8); te = G.yr == yr
        m = LogisticRegression(C=1, max_iter=300).fit(G.loc[tr, c], G.y[tr]); p[te] = m.predict_proba(G.loc[te, c])[:, 1]
    res[k] = p
T = G[G.yr >= 2018]
print(tour, pd.DataFrame({lv: {k: round(ll(v[z.index], z.y), 4) for k, v in res.items()} for lv, z in T.groupby('lv')}).to_string(), flush=True)
