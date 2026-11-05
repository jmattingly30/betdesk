"""Matchup-specific ratings, tested one at a time on 2018-24 (walk-forward, yearly refit on prior 8 years).
Base = Elo + surface Elo + games-won rating (+ newness). Candidates:
  surf_g : games-won rating by surface (player x surface), on top of the overall one
  style  : each player's games-share residual vs opponent STYLE class (big server / middle / grinder, from the
           opponent's past tiebreak rate and games per set), shrunk by count
  style_s: the same by surface (style class x surface)
  hand   : residual vs left-handers / right-handers
  tbx    : edge compression when two big servers meet (games-rating gap x combined tiebreak rate)
Usage: matchup.py atp|wta"""
import sys, re, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
tour = sys.argv[1]
F = pd.read_parquet(f'../data/feat_{tour}.parquet', columns=['date', 'level', 'surface', 'a', 'b', 'y', 'ret', 'score', 'elo_a', 'elo_b', 'elos_a', 'elos_b', 'n_a', 'n_b', 'games_w', 'games_l', 'hand_a', 'hand_b'])
F = F[F.date >= '2008-01-01'].reset_index(drop=True)
N = len(F)
A, B, Y, R, S = F.a.values, F.b.values, F.y.values, F.ret.values, F.surface.fillna('Hard').values
GW, GL, SC = F.games_w.values, F.games_l.values, F.score.fillna('').values
HA, HB = F.hand_a.fillna('U').values, F.hand_b.fillna('U').values
def sets_of(sc):
    out = []
    for m in re.finditer(r'(\d+)-(\d+)', sc):
        x, y = int(m.group(1)), int(m.group(2))
        if x + y >= 6 or max(x, y) >= 6: out.append((x, y))
    return out
g = defaultdict(float); ng = defaultdict(int)          # overall games rating
gs = defaultdict(float); ngs = defaultdict(int)        # by surface
tbr = defaultdict(lambda: [0.0, 0.0])                  # player -> [tiebreak sets, sets] (decayed)
gpsr = defaultdict(lambda: [0.0, 0.0])                 # player -> [games, sets]
rs = defaultdict(lambda: [0.0, 0.0])                   # (player, class) -> [resid sum, count]
rss = defaultdict(lambda: [0.0, 0.0])                  # (player, class, surface)
rh = defaultdict(lambda: [0.0, 0.0])                   # (player, hand of opp)
cols = {k: np.zeros(N) for k in ('gelo', 'gelo_n', 'surf_g', 'style', 'style_s', 'hand', 'tb_sum')}
def kfac(n): return 3.6 / (min(n, 60) + 8) ** 0.5
def style(p):
    t, s = tbr[p]
    if s < 8: return 'u'
    r = t / s; gp = gpsr[p][0] / max(gpsr[p][1], 1)
    if r > 0.22 or gp > 10.3: return 'big'
    if r < 0.10: return 'grind'
    return 'mid'
def shr(v, k=8): return v[0] / (v[1] + k)
for i in range(N):
    a, b, s = A[i], B[i], S[i]
    ca, cb = style(a), style(b)
    cols['gelo'][i] = g[a] - g[b]; cols['gelo_n'][i] = min(ng[a], ng[b])
    cols['surf_g'][i] = gs[(a, s)] - gs[(b, s)]
    cols['style'][i] = shr(rs[(a, cb)]) - shr(rs[(b, ca)])
    cols['style_s'][i] = shr(rss[(a, cb, s)]) - shr(rss[(b, ca, s)])
    cols['hand'][i] = shr(rh[(a, HB[i])]) - shr(rh[(b, HA[i])])
    ta, tb_ = tbr[a], tbr[b]
    cols['tb_sum'][i] = (ta[0] / (ta[1] + 5)) + (tb_[0] / (tb_[1] + 5))
    tot = GW[i] + GL[i]
    if R[i] or not (tot == tot and tot > 5): continue
    sa = (GW[i] if Y[i] == 1 else GL[i]) / tot
    e = 1 / (1 + np.exp(-(g[a] - g[b]))); res = sa - e
    g[a] += kfac(ng[a]) * res; g[b] -= kfac(ng[b]) * res; ng[a] += 1; ng[b] += 1
    es = 1 / (1 + np.exp(-(gs[(a, s)] - gs[(b, s)]))); rs_ = sa - es
    gs[(a, s)] += kfac(ngs[(a, s)]) * rs_; gs[(b, s)] -= kfac(ngs[(b, s)]) * rs_; ngs[(a, s)] += 1; ngs[(b, s)] += 1
    for p, q, cq, sg in ((a, b, cb, 1), (b, a, ca, -1)):
        for D, key in ((rs, (p, cq)), (rss, (p, cq, s)), (rh, (p, HB[i] if sg == 1 else HA[i]))):
            v = D[key]; v[0] = 0.97 * v[0] + sg * res; v[1] = 0.97 * v[1] + 1
    st = sets_of(SC[i])
    if st:
        ntb = sum(1 for x, y in st if {x, y} == {6, 7}); ngm = sum(x + y for x, y in st)
        for p in (a, b):
            tbr[p][0] = 0.98 * tbr[p][0] + ntb; tbr[p][1] = 0.98 * tbr[p][1] + len(st)
            gpsr[p][0] = 0.98 * gpsr[p][0] + ngm; gpsr[p][1] = 0.98 * gpsr[p][1] + len(st)
for k, v in cols.items(): F[k] = v
F['elo'] = lg(1 / (1 + 10 ** ((F.elo_b - F.elo_a) / 400))); F['elos'] = lg(1 / (1 + 10 ** ((F.elos_b - F.elos_a) / 400)))
F['gelo_x_new'] = F.gelo / np.sqrt(1 + F.gelo_n / 20)
F['tbx'] = F.gelo * F.tb_sum
lvmap = {'Grand Slams': 'Slam', 'Masters': 'Tour', '250 or 500': 'Tour', 'Challenger': 'Challenger', 'Futures': 'ITF', 'Premier': 'Tour',
         'Premier Mandatory': 'Tour', 'International': 'Tour', 'C10': 'ITF', 'C25': 'ITF', 'C50': 'ITF', 'C75': 'ITF', 'C100': 'ITF', 'Olympics': 'Tour', 'Tour Finals': 'Tour'}
F['lv'] = F.level.map(lvmap).fillna('Tour'); F['yr'] = F.date.dt.year
use = (F.n_a >= 5) & (F.n_b >= 5) & ~F.ret
G = F[use]
F[['date', 'a', 'b', 'surf_g', 'style', 'style_s', 'hand', 'tbx', 'tb_sum']].to_parquet(f'../data/matchup_{tour}.parquet')
if len(sys.argv) > 2: sys.exit()
BASE = ['elo', 'elos', 'gelo', 'gelo_x_new']
VARS = {'base': BASE, '+surf_g': BASE + ['surf_g'], '+style': BASE + ['style'], '+style_s': BASE + ['style_s'], '+hand': BASE + ['hand'],
        '+tbx': BASE + ['tbx'], '+all': BASE + ['surf_g', 'style', 'style_s', 'hand', 'tbx']}
res = {}; coef = {}
for k, c in VARS.items():
    p = pd.Series(np.nan, G.index)
    for yr in range(2018, 2025):
        tr = (G.yr < yr) & (G.yr >= yr - 8); te = G.yr == yr
        Xt = pd.concat([G.loc[tr, c], -G.loc[tr, c]]); yt = np.r_[G.y[tr], 1 - G.y[tr]]
        m = LogisticRegression(C=1, fit_intercept=False, max_iter=300).fit(Xt, yt); p[te] = m.predict_proba(G.loc[te, c])[:, 1]
    res[k] = p; coef[k] = dict(zip(c, m.coef_[0].round(3)))
T = G[G.yr >= 2018]
out = pd.DataFrame({lv: {k: ll(v[z.index], z.y) for k, v in res.items()} for lv, z in T.groupby('lv')})
out = out - out.loc['base']; out.loc['base'] = pd.DataFrame({lv: {'base': ll(res['base'][z.index], z.y)} for lv, z in T.groupby('lv')}).loc['base']
print(tour, '(base row = log loss; other rows = change vs base, negative = better)'); print(out.round(4).to_string())
print('coefs +all', coef['+all'])
print('style classes (share of match slots):', pd.Series([1]).shape)
