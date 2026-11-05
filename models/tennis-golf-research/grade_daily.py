"""Grade daily pricer flags: join each flag to the next-day tennisexplorer result (winner + closing price).
CLV = closing fair price of our side x the price we flagged - 1. Prints per-day and running totals."""
import glob, json, os
import numpy as np, pandas as pd
from bridge import short_key
from te_vs_mkt import fair
R = pd.read_csv('../../tex/results_all.csv')
R['ka'] = R.pa.map(short_key); R['kb'] = R.pb.map(short_key)
ok = (R.odds_a > 1) & (R.odds_b > 1)
R['fa'] = np.where(ok, fair(R.odds_a.fillna(2).values, R.odds_b.fillna(2).values), np.nan)
R['wa'] = pd.to_numeric(R.sets_a, errors='coerce') > pd.to_numeric(R.sets_b, errors='coerce')
idx = {}
for r in R.itertuples(index=False):
    idx[(r.ka, r.kb, r.day)] = (r.wa, r.fa); idx[(r.kb, r.ka, r.day)] = (not r.wa, 1 - r.fa if r.fa == r.fa else np.nan)
rows = []
for f in sorted(glob.glob('../../tex/daily_*.json')):
    day = os.path.basename(f)[6:14]; d = f'{day[:4]}-{day[4:6]}-{day[6:]}'
    for p in json.load(open(f)):
        hit = idx.get((short_key(p['player']), short_key(p['opponent']), d))
        rows.append(dict(day=d, player=p['player'], price=p['price'], edge=p['edge'], won=hit[0] if hit else None, close=hit[1] if hit else np.nan))
G = pd.DataFrame(rows)
G['ret'] = np.where(G.won.isna(), np.nan, np.where(G.won == True, G.price - 1, -1.0))
G['clv'] = G.close * G.price - 1
print(G.to_string(index=False))
print('graded', G.ret.notna().sum(), 'of', len(G), 'ROI', round(G.ret.mean(), 3), 'CLV', round(G.clv.mean(), 3))
G.to_csv('../../tex/daily_graded.csv', index=False)
