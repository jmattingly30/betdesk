"""J's totals "accuracy check" for the NFL (asked 2026-10-10 for CFB, now NFL 2026-10-11): does a tempo x efficiency
total (how many plays/drives the two teams will run x how many points each gets per play/drive) beat or add to the
summed team-points projection the app uses? Plus weather (wind, cold, dome) and blowout terms.
All ratings walk forward (each game is predicted from earlier games only); weights tuned 2008-2015, tested 2016-2025
against the closing total.
usage: python3 nfl/totals_check.py                         -> nfl/totals_check_log.txt (stdout)"""
import os, numpy as np, pandas as pd, statsmodels.api as sm
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tg = pd.read_csv(f'{H}/data/nfl_team_games.csv')
gm = pd.read_parquet(f'{H}/data/nfl_games_features.parquet')
gm = gm[gm.margin.notna() & gm.season.between(2006, 2025)].copy()
gm['date'] = pd.to_datetime(gm.gameday); gm = gm.sort_values(['date', 'gametime', 'game_id']).reset_index(drop=True)
T = tg.set_index(['game_id', 'team'])
for c in ['plays', 'drives', 'epa', 'pf', 'start_y100', 'ng_epa']:
    gm['h_' + c] = T[c].reindex(pd.MultiIndex.from_arrays([gm.game_id, gm.home_team])).values
    gm['a_' + c] = T[c].reindex(pd.MultiIndex.from_arrays([gm.game_id, gm.away_team])).values
gm = gm[gm.h_plays.notna() & gm.a_plays.notna()].reset_index(drop=True)
gm['h_ppd'] = gm.home_score / gm.h_drives; gm['a_ppd'] = gm.away_score / gm.a_drives
gm['h_ppp'] = gm.home_score / gm.h_plays; gm['a_ppp'] = gm.away_score / gm.a_plays

def od(col_h, col_a, k, carry, mu0):
    """walk-forward stat = mu + O[team] + D[opp]; returns pre-game expectations for home and away offense."""
    O, D = {}, {}; mu = mu0; last = None; eh = np.zeros(len(gm)); ea = np.zeros(len(gm))
    for i, r in enumerate(gm[['season', 'home_team', 'away_team', col_h, col_a]].itertuples(index=False)):
        s, h, a, yh, ya = r
        if s != last: O = {t: carry * v for t, v in O.items()}; D = {t: carry * v for t, v in D.items()}; last = s
        ph = mu + O.get(h, 0) + D.get(a, 0); pa = mu + O.get(a, 0) + D.get(h, 0); eh[i], ea[i] = ph, pa
        e1, e2 = yh - ph, ya - pa
        O[h] = O.get(h, 0) + k * e1; D[a] = D.get(a, 0) + k * e1; O[a] = O.get(a, 0) + k * e2; D[h] = D.get(h, 0) + k * e2
        mu += 0.002 * (e1 + e2) / 2
    return eh, ea

def pace(col, k, carry, mu0):
    """walk-forward game-level count (both teams' plays or drives) = mu + P[home] + P[away]."""
    P = {}; mu = mu0; last = None; e = np.zeros(len(gm))
    for i, r in enumerate(gm[['season', 'home_team', 'away_team', col]].itertuples(index=False)):
        s, h, a, y = r
        if s != last: P = {t: carry * v for t, v in P.items()}; last = s
        p = mu + P.get(h, 0) + P.get(a, 0); e[i] = p; err = y - p
        P[h] = P.get(h, 0) + k * err; P[a] = P.get(a, 0) + k * err; mu += 0.002 * err
    return e

gm['plays_g'] = gm.h_plays + gm.a_plays; gm['drives_g'] = gm.h_drives + gm.a_drives
tune = gm.season.between(2008, 2015).values; test = gm.season.between(2016, 2025).values & gm.total_line.notna().values
def best(fn, target, grid):
    b = None
    for k in grid[0]:
        for c in grid[1]:
            p = fn(k, c); mse = np.mean((target[tune] - p[tune]) ** 2)
            if b is None or mse < b[0]: b = (mse, k, c, p)
    return b
G = ([0.02, 0.04, 0.06, 0.08, 0.1], [0.3, 0.5, 0.7])
_, k, c, _ = best(lambda k, c: sum(od('home_score', 'away_score', k, c, 21.5)), gm.total.values, G); ph, pa = od('home_score', 'away_score', k, c, 21.5)
gm['A_sum'] = ph + pa; print(f'A summed team points: k {k} carry {c}')
_, k, c, p = best(lambda k, c: pace('plays_g', k, c, 128), gm.plays_g.values, G); gm['plays_e'] = p; print(f'  plays pace: k {k} carry {c}')
_, k, c, p = best(lambda k, c: pace('drives_g', k, c, 23), gm.drives_g.values, G); gm['drives_e'] = p; print(f'  drives pace: k {k} carry {c}')
_, k, c, _ = best(lambda k, c: sum(od('h_ppp', 'a_ppp', k, c, 0.33)), (gm.h_ppp + gm.a_ppp).values, G); eh, ea = od('h_ppp', 'a_ppp', k, c, 0.33)
gm['B_tempo_ppp'] = gm.plays_e / 2 * (eh + ea); print(f'B plays x points per play: k {k} carry {c}')
_, k, c, _ = best(lambda k, c: sum(od('h_ppd', 'a_ppd', k, c, 1.9)), (gm.h_ppd + gm.a_ppd).values, G); eh, ea = od('h_ppd', 'a_ppd', k, c, 1.9)
gm['C_drives_ppd'] = gm.drives_e / 2 * (eh + ea); print(f'C drives x points per drive: k {k} carry {c}')
_, k, c, _ = best(lambda k, c: sum(od('h_epa', 'a_epa', k, c, 0.0)), (gm.h_epa + gm.a_epa).values, G); eh, ea = od('h_epa', 'a_epa', k, c, 0.0)
gm['epa_sum'] = eh + ea
# D: EPA version: points = plays x (a + b x EPA/play), a, b fit on the tuning years
X = sm.add_constant(np.column_stack([gm.plays_e, gm.plays_e * gm.epa_sum]))
fit = sm.OLS(gm.total[tune], X[tune]).fit(); gm['D_tempo_epa'] = fit.predict(X); print('D plays x EPA/play (fit 2008-15):', fit.params.round(3).tolist())
gm['dome'] = gm.roof.isin(['dome', 'closed']).astype(int)
gm['wind'] = np.where(gm.dome == 1, 0, gm.wind.fillna(gm.wind.median()))
gm['w10'] = ((gm.wind >= 10) & (gm.wind < 20)).astype(int); gm['w20'] = (gm.wind >= 20).astype(int)
gm['cold'] = ((gm.dome == 0) & (gm.temp.fillna(60) <= 32)).astype(int)
gm['spread_abs'] = gm.spread_line.abs()
t = gm[test]
print(f'\nTest 2016-2025, {len(t)} games with a closing total. RMSE vs actual total (closing line {np.sqrt(np.mean((t.total - t.total_line) ** 2)):.2f}):')
M = ['A_sum', 'B_tempo_ppp', 'C_drives_ppd', 'D_tempo_epa']
for m in M:
    print(f'  {m:14s} RMSE {np.sqrt(np.mean((t.total - t[m]) ** 2)):.2f}  bias {np.mean(t[m] - t.total):+.2f}  corr with line {np.corrcoef(t[m], t.total_line)[0, 1]:.2f}')
print('\nDoes each add anything beyond the closing total? (actual - line) ~ (model - line), test years:')
for m in M:
    r = sm.OLS(t.total - t.total_line, sm.add_constant(t[m] - t.total_line)).fit()
    big = (t[m] - t.total_line).abs() >= 3; side = np.sign(t[m] - t.total_line)[big]; res = np.sign(t.total - t.total_line)[big]
    hit = np.mean((side == res)[res != 0])
    print(f'  {m:14s} coef {r.params.iloc[1]:+.3f} (t {r.tvalues.iloc[1]:+.2f})   3+ pt disagreements: n {big.sum():4d}, right side {hit:.1%}')
print('\nStack on the line (fit on test years, so read t-stats not fit): actual - line ~ every model gap + weather + spread size')
gaps = pd.DataFrame({m: t[m] - t.total_line for m in M})
Z = pd.concat([gaps, t[['w10', 'w20', 'cold', 'dome', 'spread_abs']]], axis=1)
print(sm.OLS(t.total - t.total_line, sm.add_constant(Z)).fit().summary().tables[1])
print('\nWeather vs the closing total, test years (over rate / actual - line):')
for nm, msk in [('wind 10-19 mph', t.w10 == 1), ('wind 20+', t.w20 == 1), ('cold <=32F', t.cold == 1), ('dome', t.dome == 1)]:
    s = t[msk]; print(f'  {nm:15s} n {len(s):4d}  over {np.mean(s.total > s.total_line):.1%}  actual - line {np.mean(s.total - s.total_line):+.2f}')
print('\nAll 2006-2025 wind 10-19 mph games by era (over rate):')
for a, z in [(2006, 2012), (2013, 2017), (2018, 2025)]:
    s = gm[gm.season.between(a, z) & (gm.w10 == 1) & gm.total_line.notna()]; print(f'  {a}-{z}: n {len(s)}  over {np.mean(s.total > s.total_line):.1%}  actual - line {np.mean(s.total - s.total_line):+.2f}')
gm[['game_id', 'season', 'total', 'total_line'] + M + ['plays_e', 'drives_e', 'w10', 'w20', 'cold', 'dome']].to_parquet('/tmp/claude-0/nflwork/totals_check.parquet')
