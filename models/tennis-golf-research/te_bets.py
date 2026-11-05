"""Betting check of the opener edge: blend (market open + our model, fit on prior 4 years) vs bet365 OPENING price.
Bet when edge > X. Report ROI vs results and CLV vs the closing fair price, by level, 2018-24."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
from te_vs_mkt import lg, fair, LV  # noqa (runs nothing heavy? te_vs_mkt runs at import) 
rows = []
for tour in ('atp', 'wta'):
    T = pd.read_parquet(f'../data/te_{tour}.parquet')
    T = T[T.completed.isin(['completed', 'retired_or_abandoned']) & T.p_model.notna()].copy()
    T['void'] = (T.completed != 'completed') & (T.total_games < 6)  # bet365: void unless a set is done (approx: <6 games played)
    T['lv'] = T.level.map(LV); T = T[T.lv.notna()]; T['yr'] = T.date.dt.year
    for sfx, nm in (('', 'close'), ('_opening', 'open')):
        pb = fair(T[f'Betfair_a{sfx}'].values, T[f'Betfair_b{sfx}'].values); p3 = fair(T[f'bet365_a{sfx}'].values, T[f'bet365_b{sfx}'].values)
        ok = (T[f'Betfair_a{sfx}'] > 1) & (T[f'Betfair_b{sfx}'] > 1) & (1 / T[f'Betfair_a{sfx}'] + 1 / T[f'Betfair_b{sfx}'] < 1.08)
        T['m_' + nm] = np.where(ok, pb, p3)
    T = T[T.m_close.between(0.01, 0.99) & T.m_open.between(0.01, 0.99) & (T.bet365_a_opening > 1) & (T.bet365_b_opening > 1)]
    for lv, z in T.groupby('lv'):
        for yr in range(2018, 2025):
            tr, te = z[(z.yr < yr) & (z.yr >= yr - 4)], z[z.yr == yr]
            if len(te) == 0 or len(tr) < 500: continue
            trc = tr[tr.completed == 'completed']
            X = np.c_[lg(trc.m_open), lg(trc.p_model)]
            lr = LogisticRegression(C=10).fit(np.r_[X, -X], np.r_[trc.y, 1 - trc.y])
            lr0 = LogisticRegression(C=10).fit(np.r_[X[:, :1], -X[:, :1]], np.r_[trc.y, 1 - trc.y])
            for kind in ('model+market', 'market only'):
              p = lr.predict_proba(np.c_[lg(te.m_open), lg(te.p_model)])[:, 1] if kind == 'model+market' else lr0.predict_proba(np.c_[lg(te.m_open)])[:, 1]
              for side in (1, 0):
                q = p if side else 1 - p
                pr = (te.bet365_a_opening if side else te.bet365_b_opening).values
                won = (te.y == side).values
                cl = (te.m_close if side else 1 - te.m_close).values
                rows.append(pd.DataFrame(dict(kind=kind, tour=tour, lv=lv, yr=yr, edge=q * pr - 1, price=pr, ret=np.where(te.void.values, 0.0, np.where(won, pr - 1, -1.0)), clv=cl * pr - 1)))
B = pd.concat(rows)
B = B[B.price.between(1.2, 5)]
out = []
for (kind, tour, lv), z in B.groupby(['kind', 'tour', 'lv']):
    for th in (0.03, 0.06):
        w = z[z.edge > th]
        out.append(dict(kind=kind, tour=tour, level=lv, edge_over=th, bets=len(w), per_year=round(len(w) / z.yr.nunique()), avg_edge=w.edge.mean(), clv=w.clv.mean(),
                        roi=w.ret.mean(), roi_se=w.ret.std() / np.sqrt(max(len(w), 1)), yrs_pos=(w.groupby('yr').ret.mean() > 0).sum(), yrs=w.yr.nunique()))
O = pd.DataFrame(out); print(O.round(4).to_string()); O.to_csv('../../op/te_bets.csv', index=False)
