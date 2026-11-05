"""Opener test with a market rating: each player's rating from PAST closing prices (Kalman on logits, bet365/Betfair
close), added to the opener+model blend. Same setup as te_bets.py (bet365 opening price, prior-4-yr fits, voids)."""
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
from te_vs_mkt import lg, fair, LV

def mkt(T, q=0.003, obs=0.05):
    m = defaultdict(float); v = defaultdict(lambda: 1.0); last = {}
    out = np.zeros((len(T), 2))
    for i, r in enumerate(T[['a', 'b', 'date', 'm_close']].itertuples(index=False)):
        a, b, d = r.a, r.b, r.date
        for p in (a, b):
            if p in last: v[p] = min(1.0, v[p] + q * (d - last[p]).days)
        out[i] = (m[a] - m[b], 1 - max(v[a], v[b]))
        if r.m_close == r.m_close:
            res = lg(r.m_close) - (m[a] - m[b]); S = v[a] + v[b] + obs
            m[a] += v[a] / S * res; m[b] -= v[b] / S * res
            v[a] *= 1 - v[a] / S; v[b] *= 1 - v[b] / S
        last[a] = last[b] = d
    return out

rows = []; LL = []
for tour in ('atp', 'wta'):
    T = pd.read_parquet(f'../data/te_{tour}.parquet')
    T = T[T.completed.isin(['completed', 'retired_or_abandoned']) & T.p_model.notna()].copy()
    T['void'] = (T.completed != 'completed') & (T.total_games < 6)
    T['lv'] = T.level.map(LV); T['yr'] = T.date.dt.year
    for sfx, nm in (('', 'close'), ('_opening', 'open')):
        pb = fair(T[f'Betfair_a{sfx}'].values, T[f'Betfair_b{sfx}'].values); p3 = fair(T[f'bet365_a{sfx}'].values, T[f'bet365_b{sfx}'].values)
        ok = (T[f'Betfair_a{sfx}'] > 1) & (T[f'Betfair_b{sfx}'] > 1) & (1 / T[f'Betfair_a{sfx}'] + 1 / T[f'Betfair_b{sfx}'] < 1.08)
        T['m_' + nm] = np.where(ok, pb, p3)
    T.loc[~T.m_close.between(0.01, 0.99), 'm_close'] = np.nan
    T = T.sort_values(['date', 'rnd'])
    M = mkt(T); T['mkx'] = M[:, 0] * np.clip(M[:, 1], 0, 1)
    T = T[T.lv.notna() & T.m_close.notna() & T.m_open.between(0.01, 0.99) & (T.bet365_a_opening > 1) & (T.bet365_b_opening > 1)]
    for lv, z in T.groupby('lv'):
        for yr in range(2018, 2025):
            tr, te = z[(z.yr < yr) & (z.yr >= yr - 4)], z[z.yr == yr]
            if len(te) == 0 or len(tr) < 500: continue
            trc = tr[tr.completed == 'completed']
            def X(d, k): return np.column_stack([[lg(d.m_open), lg(d.p_model), d.mkx.values][i] for i in k])
            for kind, k in (('open+model', [0, 1]), ('open+model+mkt', [0, 1, 2]), ('open+mkt', [0, 2])):
                A = X(trc, k); lr = LogisticRegression(C=10).fit(np.r_[A, -A], np.r_[trc.y, 1 - trc.y])
                p = lr.predict_proba(X(te, k))[:, 1]
                tc = te.completed.values == 'completed'
                LL.append(dict(kind=kind, tour=tour, lv=lv, yr=yr, n=tc.sum(), ll=-np.mean(np.where(te.y[tc] == 1, np.log(p[tc]), np.log(1 - p[tc]))),
                               ll_open=-np.mean(np.where(te.y[tc] == 1, np.log(te.m_open[tc]), np.log(1 - te.m_open[tc]))),
                               ll_close=-np.mean(np.where(te.y[tc] == 1, np.log(te.m_close[tc]), np.log(1 - te.m_close[tc])))))
                for side in (1, 0):
                    q = p if side else 1 - p
                    pr = (te.bet365_a_opening if side else te.bet365_b_opening).values
                    won = (te.y == side).values
                    cl = (te.m_close if side else 1 - te.m_close).values
                    rows.append(pd.DataFrame(dict(kind=kind, tour=tour, lv=lv, yr=yr, edge=q * pr - 1, price=pr, ret=np.where(te.void.values, 0.0, np.where(won, pr - 1, -1.0)), clv=cl * pr - 1)))
L = pd.DataFrame(LL)
L = L.assign(w_ll=L.ll * L.n, w_o=L.ll_open * L.n, w_c=L.ll_close * L.n).groupby(['tour', 'lv', 'kind'])[['n', 'w_ll', 'w_o', 'w_c']].sum()
L = pd.DataFrame(dict(n=L.n, ll=L.w_ll / L.n, open=L.w_o / L.n, close=L.w_c / L.n)); print(L.round(4).to_string())
B = pd.concat(rows); B = B[B.price.between(1.2, 5)]
out = []
for (tour, lv, kind), z in B.groupby(['tour', 'lv', 'kind']):
    for th in (0.03, 0.06):
        w = z[z.edge > th]
        out.append(dict(tour=tour, level=lv, kind=kind, edge_over=th, bets=len(w), per_year=round(len(w) / z.yr.nunique()), clv=w.clv.mean(),
                        roi=w.ret.mean(), roi_se=w.ret.std() / np.sqrt(max(len(w), 1)), yrs_pos=(w.groupby('yr').ret.mean() > 0).sum(), yrs=w.yr.nunique()))
O = pd.DataFrame(out); print(O.round(4).to_string()); O.to_csv('../../op/te_mkt.csv', index=False)
