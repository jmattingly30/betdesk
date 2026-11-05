"""Can a games-won rating stand in for serve/return point ratings when only scores are available (2025-26)?
Walk-forward 'game Elo': expected game share from rating gap, updated by actual share of games won. Tested 2018-24."""
import numpy as np, pandas as pd, sys
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
tour = sys.argv[1] if len(sys.argv) > 1 else 'atp'
F = pd.read_parquet(f'../data/feat_{tour}.parquet', columns=['date', 'level', 'a', 'b', 'y', 'ret', 'elo_a', 'elo_b', 'elos_a', 'elos_b', 'n_a', 'n_b', 'p_pt', 'games_w', 'games_l'])
g = defaultdict(float); ng = defaultdict(int)
ga = np.zeros(len(F)); gn = np.zeros(len(F))
for i, r in enumerate(F.itertuples(index=False)):
    a, b = r.a, r.b
    ga[i] = g[a] - g[b]; gn[i] = min(ng[a], ng[b])
    tot = r.games_w + r.games_l
    if tot == tot and tot > 5 and not r.ret:
        sa = (r.games_w if r.y == 1 else r.games_l) / tot
        e = 1 / (1 + np.exp(-(g[a] - g[b])))
        k = 0.9 / (min(ng[a], 60) + 8) ** 0.5
        g[a] += k * (sa - e) * 4; g[b] -= k * (sa - e) * 4
        ng[a] += 1; ng[b] += 1
F['gelo'] = ga
F['gelo_n'] = gn
F[['gelo', 'gelo_n']].to_parquet(f'../data/gelo_{tour}.parquet')
F['elo'] = lg(1 / (1 + 10 ** ((F.elo_b - F.elo_a) / 400))); F['elos'] = lg(1 / (1 + 10 ** ((F.elos_b - F.elos_a) / 400)))
F['pt'] = lg(F.p_pt.fillna(0.5))
F = F[(F.n_a >= 5) & (F.n_b >= 5) & ~F.ret]
F['yr'] = F.date.dt.year
lvmap = {'Grand Slams': 'Slam', 'Masters': 'Tour', '250 or 500': 'Tour', 'Challenger': 'Challenger', 'Futures': 'ITF',
         'Premier': 'Tour', 'Premier Mandatory': 'Tour', 'International': 'Tour', 'C10': 'ITF', 'C25': 'ITF', 'C50': 'ITF', 'C75': 'ITF', 'C100': 'ITF'}
F['lv'] = F.level.map(lvmap)
res = {}
for nm, cols in (('elo', ['elo', 'elos']), ('elo+games', ['elo', 'elos', 'gelo']), ('elo+points', ['elo', 'elos', 'pt']), ('elo+points+games', ['elo', 'elos', 'pt', 'gelo'])):
    p = pd.Series(np.nan, F.index)
    for yr in range(2018, 2025):
        tr = (F.yr < yr) & (F.yr >= yr - 8); te = F.yr == yr
        m = LogisticRegression(C=1, max_iter=300).fit(F.loc[tr, cols], F.y[tr]); p[te] = m.predict_proba(F.loc[te, cols])[:, 1]
    res[nm] = p
T = F[F.yr >= 2018]
print(tour, pd.DataFrame({lv: {k: round(ll(v[z.index], z.y), 4) for k, v in res.items()} for lv, z in T.groupby('lv')}).to_string())
