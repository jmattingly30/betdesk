"""Grand Slam test vs bookmaker odds (tennis-data.co.uk columns via deuce)."""
import sys
import numpy as np, pandas as pd
import odds
from model import ll, lgt
from sklearn.linear_model import LogisticRegression

tour = sys.argv[1] if len(sys.argv) > 1 else 'atp'
F = pd.read_parquet(f'../data/feat_{tour}.parquet')
P = pd.read_parquet(f'../data/pred_{tour}.parquet')
F = F.join(P)
J = odds.join(F, tour)
S = F.loc[J.index].join(J)
print('joined', len(S), 'of odds rows', len(pd.read_parquet(f'../data/{tour}_odds.parquet')))
# A-perspective prices
aw = S.y == 1
for bk in ['PS', 'B365', 'Max', 'Avg']:
    S[bk + 'a'] = np.where(aw, S[bk + 'W'], S[bk + 'L']).astype(float)
    S[bk + 'b'] = np.where(aw, S[bk + 'L'], S[bk + 'W']).astype(float)
S = S[(S.PSa > 1) & (S.PSb > 1) & (~S.ret)].copy()
# drop other-book prices that look like data errors (more than 15% away from Pinnacle on the same side)
for bk in ['B365', 'Max', 'Avg']:
    for sd in 'ab':
        bad = (np.abs(np.log(S[bk + sd] / S['PS' + sd])) > 0.15)
        S.loc[bad, bk + sd] = np.nan
ia, ib = 1 / S.PSa, 1 / S.PSb
S['hold'] = ia + ib - 1
S['p_mkt'] = ia / (ia + ib)
# power devig (accounts for longshot margin)
def power(a, b):
    lo, hi = np.ones_like(a), np.full_like(a, 3.0)
    for _ in range(60):
        k = (lo + hi) / 2
        s = a ** k + b ** k
        hi = np.where(s < 1, k, hi); lo = np.where(s >= 1, k, lo)
    return a ** k
S['p_mkt_pow'] = power(ia.values, ib.values)
S['yr'] = S.date.dt.year
S = S[S.yr >= 2010]
print('Pinnacle rows', len(S), 'years', S.yr.min(), S.yr.max(), 'avg hold', round(S.hold.mean(), 4))
y = S.y.values
for c in ['p_mkt', 'p_mkt_pow', 'p_elo', 'p_eb', 'p_pt', 'p_base', 'p_model']:
    z = S[c].notna()
    print(f'{c:10s} n={z.sum()} logloss={ll(S[c][z], y[z]):.4f} acc={((S[c][z] > .5) == (y[z] == 1)).mean():.4f}')
# market + model blend, walk forward by year
S['l_mkt'] = lgt(S.p_mkt_pow); S['l_mod'] = lgt(S.p_model)
S['p_blend'] = np.nan
coefs = []
for t in range(2013, 2024):
    tr = (S.yr < t) & S.p_model.notna(); te = (S.yr == t) & S.p_model.notna()
    if te.sum() == 0:
        continue
    m = LogisticRegression(C=10).fit(S.loc[tr, ['l_mkt', 'l_mod']], S.y[tr])
    S.loc[te, 'p_blend'] = m.predict_proba(S.loc[te, ['l_mkt', 'l_mod']])[:, 1]
    coefs.append((t, *np.round(m.coef_[0], 3)))
z = S.p_blend.notna()
print('blend 2013-23 n', z.sum(), 'mkt', round(ll(S.p_mkt_pow[z], y[z]), 4), 'blend', round(ll(S.p_blend[z], y[z]), 4), 'model', round(ll(S.p_model[z], y[z]), 4))
print('coefs (yr, mkt, model)', coefs[-3:])
# betting: back the side where model (or blend) beats price by edge threshold
def bets(pcol, price='PS', th=(0.0, 0.02, 0.05, 0.1)):
    res = []
    for t in th:
        r = []
        for side in ('a', 'b'):
            p = S[pcol] if side == 'a' else 1 - S[pcol]
            o = S[price + side]
            ev = p * o - 1
            won = (S.y == 1) if side == 'a' else (S.y == 0)
            sel = (ev > t) & p.notna() & o.notna()
            r.append(pd.DataFrame({'ret': np.where(won[sel], o[sel] - 1, -1.0), 'yr': S.yr[sel], 'o': o[sel]}))
        r = pd.concat(r)
        res.append({'model': pcol, 'price': price, 'edge>': t, 'bets': len(r), 'roi': round(r.ret.mean(), 4),
                    'se': round(r.ret.std() / np.sqrt(max(len(r), 1)), 4), 'avg_odds': round(r.o.mean(), 2)})
    return res
rows = []
for pc in ['p_model', 'p_blend']:
    for pr in ['PS', 'Max', 'B365']:
        rows += bets(pc, pr)
print(pd.DataFrame(rows).to_string())
# favourite-longshot check: market calibration by implied prob bucket (both sides)
L = pd.concat([pd.DataFrame({'p': S.p_mkt, 'w': S.y, 'o': S.PSa, 'om': S.Maxa}), pd.DataFrame({'p': 1 - S.p_mkt, 'w': 1 - S.y, 'o': S.PSb, 'om': S.Maxb})])
L['bk'] = pd.cut(L.p, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
print(L.groupby('bk', observed=True).apply(lambda g: pd.Series({'n': len(g), 'implied': g.p.mean(), 'actual': g.w.mean(),
      'roi_PS': (np.where(g.w == 1, g.o - 1, -1)).mean(), 'roi_Max': (np.where(g.w == 1, g.om - 1, -1)).mean()})).round(4).to_string())
fav = L[L.p > 0.5]
print('all favourites at Pinnacle: n', len(fav), 'roi', round(np.where(fav.w == 1, fav.o - 1, -1).mean(), 4))
S.to_parquet(f'../data/slam_{tour}.parquet')
