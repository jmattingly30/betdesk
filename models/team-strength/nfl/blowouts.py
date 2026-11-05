"""J 2026-10-11: are we under-projecting blowouts? How often do NFL games end 8+ apart, does our model's margin stretch
far enough, and is the normal(sigma 13.3) margin curve right for big spreads?
usage: python3 nfl/blowouts.py   (reads data/nfl_games_features.parquet; closing lines from nflverse)"""
import numpy as np, pandas as pd, os
from scipy.stats import norm
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
g = pd.read_parquet(f'{H}/data/nfl_games_features.parquet')
g = g[g.margin.notna() & g.spread_line.notna() & (g.season <= 2025)].copy()
g['fav_m'] = np.where(g.spread_line >= 0, g.margin, -g.margin); g['fav_s'] = g.spread_line.abs()
# walk-forward blend (weights fit on earlier seasons only), same form as the app
est = np.full(len(g), np.nan)
for y in range(2010, 2026):
    tr, te = (g.season < y).values, (g.season == y).values
    X = np.column_stack([np.ones(tr.sum()), g.rt_pred[tr], g.mkt_pred[tr]]); b = np.linalg.lstsq(X, g.margin[tr], rcond=None)[0]
    est[te] = b[0] + b[1] * g.rt_pred[te] + b[2] * g.mkt_pred[te]
g['est'] = est; g = g[g.season >= 2010]
print(f'NFL 2010-2025 regular+post, {len(g)} games\n')
print('1. How often do games end 8+ points apart?')
print(f'  final margin 8+: {np.mean(g.margin.abs() >= 8):.1%}   14+: {np.mean(g.margin.abs() >= 14):.1%}   21+: {np.mean(g.margin.abs() >= 21):.1%}')
print(f'  closing spread 7.5+: {np.mean(g.fav_s >= 7.5):.1%}   10+: {np.mean(g.fav_s >= 10):.1%}')
print(f'  our model margin 7.5+: {np.mean(np.abs(g.est) >= 7.5):.1%}   10+: {np.mean(np.abs(g.est) >= 10):.1%}')
print(f'  spread of projections (SD): closing line {g.spread_line.std():.2f}, our model {g.est.std():.2f}, score rating alone {g.rt_pred.std():.2f}\n')
print('2. Does the model stretch far enough? Slope of actual margin on the projection (1.0 = right size, >1 = too timid)')
for c in ['est', 'spread_line', 'rt_pred', 'mkt_pred']:
    b = np.polyfit(g[c], g.margin, 1); print(f'  {c:12s} slope {b[0]:.2f}  RMSE {np.sqrt(np.mean((g.margin - g[c]) ** 2)):.2f}')
print('  by season block (walk-forward est):')
for a, z in [(2010, 2015), (2016, 2020), (2021, 2025)]:
    s = g[g.season.between(a, z)]; print(f'    {a}-{z}: slope est {np.polyfit(s.est, s.margin, 1)[0]:.2f}, line {np.polyfit(s.spread_line, s.margin, 1)[0]:.2f}')
print('\n3. By closing spread: favourite margin, cover rate, and how often the favourite wins by 8+ / 14+')
g['band'] = pd.cut(g.fav_s, [-0.1, 2.5, 3.5, 6.5, 7.5, 9.5, 13.5, 40], labels=['0-2.5', '3', '4-6.5', '7-7.5', '8-9.5', '10-13.5', '14+'])
rows = []
for b, s in g.groupby('band', observed=True):
    sd = (s.fav_m - s.fav_s).std(); cov = np.mean(s.fav_m > s.fav_s); push = np.mean(s.fav_m == s.fav_s)
    p8n = np.mean(1 - norm.cdf((7.5 - s.fav_s) / 13.3)); p14n = np.mean(1 - norm.cdf((13.5 - s.fav_s) / 13.3))
    rows.append([b, len(s), s.fav_s.mean(), s.fav_m.mean(), sd, cov, push, np.mean(s.fav_m >= 8), p8n, np.mean(s.fav_m >= 14), p14n,
                 np.mean(s.fav_m < 0), np.mean(1 - norm.cdf(s.fav_s / 13.3))])
T = pd.DataFrame(rows, columns=['spread', 'n', 'avg_spread', 'avg_fav_margin', 'sd_vs_line', 'fav_cover', 'push', 'won_by_8+', 'normal_8+',
                                 'won_by_14+', 'normal_14+', 'dog_won', 'normal_dog'])
print(T.round(3).to_string(index=False))
print('\n4. Totals and blowouts: average total vs closing total, by spread band (do lopsided games go over or under?)')
for b, s in g[g.total_line.notna()].groupby('band', observed=True):
    print(f'  {b:8s} n {len(s):4d}  actual - line {np.mean(s.total - s.total_line):+5.2f}  over {np.mean(s.total > s.total_line):.1%}  '
          f'fav pts - implied {np.mean(np.where(s.spread_line >= 0, s.home_score, s.away_score) - (s.total_line + s.fav_s) / 2):+5.2f}  '
          f'dog pts - implied {np.mean(np.where(s.spread_line >= 0, s.away_score, s.home_score) - (s.total_line - s.fav_s) / 2):+5.2f}')
print('\n5. Win-by-8+ chance: normal curve vs actual, model-projected games (walk-forward est), by projected margin')
g['eb'] = pd.cut(g.est.abs(), [-0.1, 3, 5, 7.5, 10, 40])
for b, s in g.groupby('eb', observed=True):
    fm = np.where(s.est >= 0, s.margin, -s.margin)
    print(f'  model {str(b):12s} n {len(s):4d}  avg proj {s.est.abs().mean():5.1f}  avg actual {fm.mean():5.1f}  won by 8+ {np.mean(fm >= 8):.1%} '
          f'(normal {np.mean(1 - norm.cdf((7.5 - s.est.abs()) / 13.3)):.1%})  line avg {np.mean(np.where(s.est >= 0, s.spread_line, -s.spread_line)):5.1f}')
print('\n6. Key numbers: share of games decided by exactly')
for k in [1, 2, 3, 4, 6, 7, 8, 10, 14, 17, 21]:
    print(f'  {k:2d}: {np.mean(g.margin.abs() == k):.1%}', end='')
print('\n   2021-2025 only:', ' '.join(f'{k}:{np.mean(g[g.season >= 2021].margin.abs() == k):.1%}' for k in [3, 6, 7, 10, 14]))
