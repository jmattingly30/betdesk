"""2025 holdout: the live-style model (Elo + surface + context, walk-forward by month, no point ratings) vs the 2025
market, and bets at bet365 OPENING prices when model+opener blend shows an edge. Blend weight fit on 2018-24 deuce
rows is not reused: here the blend is fit walk-forward on earlier 2025 months only."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
from model26 import walk, lg, ll
from te_vs_mkt import fair

FULL = ['elo', 'elos', 'conf', 'new_a', 'layoff', 'm21', 'n90', 'age', 'age2', 'home'] + (['gelo', 'gelo_x_new'] if len(sys.argv) > 1 else [])
te = pd.read_csv('../../kg/alimoh89_tennis-results-and-betting-odds-20142025/tennis_matches_2014_2025.csv', low_memory=False)
te = te.set_index('match_link')
out = []
allb = []
for tour in ('atp', 'wta'):
    F = pd.read_parquet(f'../data/feat25_{tour}.parquet')
    F['home'] = 0.0
    G, p, _ = walk(F, FULL)
    G['p_model'] = p
    G = G[G.p_model.notna()].copy()
    T = te.loc[G.fid]
    same = (G.a.values == T.player_a.map(lambda x: __import__('bridge').norm(x)).values)
    def col(c_a, c_b): return np.where(same, T[c_a], T[c_b]), np.where(same, T[c_b], T[c_a])
    for sfx, nm in (('', 'close'), ('_opening', 'open')):
        ba, bb = col(f'Betfair_odds_a{sfx}', f'Betfair_odds_b{sfx}'); ka, kb = col(f'bet365_odds_a{sfx}', f'bet365_odds_b{sfx}')
        ok = (ba > 1) & (bb > 1) & (1 / ba + 1 / bb < 1.08)
        G['m_' + nm] = np.where(ok, fair(ba, bb), fair(ka, kb))
        G['k_a' + nm], G['k_b' + nm] = ka, kb
    G = G[G.m_open.between(0.01, 0.99) & G.m_close.between(0.01, 0.99)]
    G['mon'] = G.start.dt.month
    for lv, z in G.groupby('cat'):
        z = z.copy(); z['blend'] = np.nan
        for m in range(4, 13):
            tr, tt = z[z.mon < m], z.mon == m
            if tt.sum() == 0 or len(tr) < 300: continue
            X = np.c_[lg(tr.m_open), lg(tr.p_model)]
            lr = LogisticRegression(C=10).fit(np.r_[X, -X], np.r_[tr.y, 1 - tr.y])
            z.loc[tt, 'blend'] = lr.predict_proba(np.c_[lg(z.m_open[tt]), lg(z.p_model[tt])])[:, 1]
        k = z.blend.notna()
        out.append(dict(tour=tour, level=lv, n=int(k.sum()), model=ll(z.p_model[k], z.y[k]), mkt_open=ll(z.m_open[k], z.y[k]), mkt_close=ll(z.m_close[k], z.y[k]),
                        blend_open=ll(z.blend[k], z.y[k])))
        zz = z[k]
        for side in (1, 0):
            q = zz.blend if side else 1 - zz.blend
            pr = zz.k_aopen if side else zz.k_bopen
            cl = zz.m_close if side else 1 - zz.m_close
            allb.append(pd.DataFrame(dict(tour=tour, lv=lv, edge=q * pr - 1, price=pr, clv=cl * pr - 1, ret=np.where(zz.y == side, pr - 1, -1.0))))
R = pd.DataFrame(out); R['gain_vs_open'] = R.mkt_open - R.blend_open
print('2025 holdout log loss (Apr-Dec, completed matches)'); print(R.round(4).to_string())
B = pd.concat(allb); B = B[B.price.between(1.2, 5)]
rows = []
for (tour, lv), z in B.groupby(['tour', 'lv']):
    for th in (0.03, 0.06):
        w = z[z.edge > th]
        rows.append(dict(tour=tour, lv=lv, edge_over=th, bets=len(w), clv=w.clv.mean(), roi=w.ret.mean(), se=w.ret.std() / np.sqrt(max(len(w), 1))))
print('\n2025 bets at bet365 opener'); print(pd.DataFrame(rows).round(4).to_string())
