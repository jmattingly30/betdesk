"""Model vs sportsbook player props, Feb 16 - Mar 8 2023 (free sample: github.com/bendominguez0111/nba-models exports/player_props).
Model is fit on 2018-19 to 2021-22 only, so 2023 is out of sample. usage: python3 props_test.py <model_dir> <props_dir>"""
import sys, glob, re, unicodedata, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); from model import fit, usable
md, pdir = sys.argv[1:3]
def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', re.sub(r'[^a-z ]', '', s)).split() and ' '.join(re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', re.sub(r'[^a-z ]', '', s)).split())
pp = pd.concat([pd.read_csv(f).assign(date=pd.to_datetime(f.split('/')[-1][:10].replace('_', '-'))) for f in sorted(glob.glob(f'{pdir}/*.csv'))])
STAT = {'player_points': 'pts', 'player_rebounds': 'reb', 'player_assists': 'ast', 'player_threes': 'fg3m', 'player_steals': 'stl',
        'player_blocks': 'blk', 'player_turnovers': 'tov'}
pp = pp[pp.prop_type.isin(STAT)].copy(); pp['stat'] = pp.prop_type.map(STAT); pp['key'] = pp.player_name.map(norm)
dec = lambda a: np.where(a > 0, 1 + a / 100, 1 + 100 / -a)
pp['imp'] = 1 / dec(pp.price.astype(float))
w = pp.pivot_table(index=['date', 'key', 'stat', 'points', 'sports_book'], columns='name', values=['imp', 'price'], aggfunc='first').dropna()
w.columns = [f'{a}_{b}' for a, b in w.columns]; w = w.reset_index()
w['p_mkt'] = w.imp_Over / (w.imp_Over + w.imp_Under)                       # no-vig over probability per book
f = pd.read_parquet(f'{md}/data/features.parquet'); f = f[usable(f)]
m = fit(f[f.season.between(2019, 2022)])
te = f[(f.date >= '2023-02-15') & (f.date <= '2023-03-09')].copy(); te['key'] = te.name.map(norm)
te = te.join(m.predict(te))
j = w.merge(te, left_on=['date', 'key'], right_on=['date', 'key'], how='inner')
print('book lines', len(w), 'matched to a played game', len(j), 'player-games', j[['date', 'key']].drop_duplicates().shape[0])
j['p_model'] = [m.prob_over(r, r.stat, r.points) for _, r in j.iterrows()]
j['actual'] = [r[r.stat] for _, r in j.iterrows()]
j = j[j.actual != j.points]                                              # drop pushes
j['over'] = (j.actual > j.points).astype(int)
ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-4, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1)))
rows = []
for s, g in list(j.groupby('stat')) + [('all', j)]:
    blend = 0.5 * g.p_model + 0.5 * g.p_mkt
    rows.append({'stat': s, 'book lines': len(g), 'over rate': g.over.mean(), 'logloss market': ll(g.p_mkt, g.over), 'logloss model': ll(g.p_model, g.over),
                 'logloss 50/50 blend': ll(blend, g.over), 'brier market': ((g.p_mkt - g.over) ** 2).mean(), 'brier model': ((g.p_model - g.over) ** 2).mean(),
                 'corr model vs market': np.corrcoef(g.p_model, g.p_mkt)[0, 1]})
res = pd.DataFrame(rows); print(res.round(4).to_string(index=False))
# betting: one bet per player-game-stat at the best price among books, when model edge vs that book's no-vig price exceeds a threshold
j['edge_over'] = j.p_model - j.imp_Over; j['edge_under'] = (1 - j.p_model) - j.imp_Under
bets = []
for thr in [0.0, 0.03, 0.05, 0.08]:
    for side in ['Over', 'Under']:
        e = j[j[f'edge_{side.lower()}'] > thr]
        e = e.sort_values(f'edge_{side.lower()}', ascending=False).drop_duplicates(['date', 'key', 'stat', 'points'])
        win = e.over == 1 if side == 'Over' else e.over == 0
        pnl = np.where(win, dec(e[f'price_{side}'].astype(float)) - 1, -1.0)
        bets.append({'edge >': thr, 'side': side, 'bets': len(e), 'win %': win.mean() if len(e) else np.nan, 'ROI': pnl.mean() if len(e) else np.nan})
bt = pd.DataFrame(bets); print(bt.round(3).to_string(index=False))
res.to_csv(f'{md}/props_2023_accuracy.csv', index=False); bt.to_csv(f'{md}/props_2023_bets.csv', index=False)
j[['date', 'name', 'team', 'opp', 'stat', 'points', 'sports_book', 'price_Over', 'price_Under', 'p_mkt', 'p_model', 'min_proj', 'actual', 'over']].to_csv(f'{md}/props_2023_lines.csv', index=False)
