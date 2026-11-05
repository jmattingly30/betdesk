"""Score engine vs bookmaker total-games and game-handicap lines, 2018-24 (te set). Engine input = OUR model only:
match win prob (p_model) + hold level from our serve/return point ratings, with the day-to-day form swing.
Compares engine P(over)/P(cover) with each book's de-vigged closing price on settled matches, and bet ROI."""
import sys
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
sys.argv = sys.argv[:1]
import op_score as E  # reuses interp/dist (bo3 tables + first-set table)
from te_vs_mkt import fair, LV

def ll(p, y): p = np.clip(p, 1e-4, 1 - 1e-4); return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
BK = ['bet365', 'Ladbrokes', 'Unibet']
rows = []
for tour in ('atp', 'wta'):
    T = pd.read_parquet(f'../data/te_{tour}.parquet')
    T = T[(T.completed == 'completed') & T.p_model.notna() & (T.date.dt.year >= 2018) & (T.bo.fillna(3) == 3) & T.pa.notna() & T.pb.notna()]
    T['lv'] = T.level.map(LV); T = T[T.lv.notna()]
    s = ((T.pa + T.pb) / 2).values; p = T.p_model.values
    d = E.dist(s, p)
    for bk in BK:
        L = T[f'{bk}_tot'].values; o, u = T[f'{bk}_over'].values, T[f'{bk}_under'].values
        ok = (L > 10) & (o > 1) & (u > 1) & np.isfinite(L)
        idx = np.where(ok)[0]
        pe = np.array([E.p_over(d['tot'][i:i + 1], L[i])[0] for i in idx])
        pf = fair(o[idx], u[idx])
        tg = T.total_games.values[idx]
        y = np.where(tg > L[idx], 1.0, np.where(tg < L[idx], 0.0, np.nan))
        rows.append(pd.DataFrame(dict(tour=tour, lv=T.lv.values[idx], book=bk, typ='total', line=L[idx], p_eng=pe, p_mkt=pf, y=y, pa=o[idx], pb=u[idx])))
        H = T[f'{bk}_hcp'].values; ha, hb = T[f'{bk}_ha'].values, T[f'{bk}_hb'].values
        ok = np.isfinite(H) & (ha > 1) & (hb > 1)
        idx = np.where(ok)[0]
        pe = np.array([E.p_cover(d['mar'][i:i + 1], H[i])[0] for i in idx])
        pf = fair(ha[idx], hb[idx])
        m = T.marA.values[idx] + H[idx]
        y = np.where(m > 0, 1.0, np.where(m < 0, 0.0, np.nan))
        rows.append(pd.DataFrame(dict(tour=tour, lv=T.lv.values[idx], book=bk, typ='handicap', line=H[idx], p_eng=pe, p_mkt=pf, y=y, pa=ha[idx], pb=hb[idx])))
R = pd.concat(rows); R = R[R.y.notna()]
R.to_parquet('../data/te_score_rows.parquet')
print('[1] Log loss on settled lines (engine from our model only vs book closing fair), plus 80/20 blend')
g = R.groupby(['tour', 'lv', 'typ'])
print(g.apply(lambda z: pd.Series(dict(n=len(z), book=ll(z.p_mkt, z.y), engine=ll(z.p_eng, z.y), blend=ll(0.8 * z.p_mkt + 0.2 * z.p_eng, z.y),
                                     eng_pred=z.p_eng.mean(), book_pred=z.p_mkt.mean(), act=z.y.mean()))).round(4).to_string())
print('\n[2] Bets where engine edge vs the price > X (both sides), ROI')
out = []
for (tour, lv, typ), z in g:
    for th in (0.03, 0.06, 0.10):
        e1 = z.p_eng * z.pa - 1; e2 = (1 - z.p_eng) * z.pb - 1
        r1 = np.where(z.y == 1, z.pa - 1, -1.0)[e1 > th]; r2 = np.where(z.y == 0, z.pb - 1, -1.0)[e2 > th]
        r = np.r_[r1, r2]
        out.append(dict(tour=tour, lv=lv, typ=typ, edge_over=th, bets=len(r), roi=r.mean() if len(r) else np.nan, se=r.std() / np.sqrt(max(len(r), 1))))
print(pd.DataFrame(out).round(4).to_string())
