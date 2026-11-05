"""Option 1, variant 2 (2026-10-10): the new caller's personnel mix -> depth players' (WR3, TE2) targets and snaps.
python3 caller_personnel_test.py <dl_dir> <model_dir>. For teams whose caller called plays elsewhere the season before:
x = (caller's old-team 11-personnel rate - this team's last-season 11 rate), +x for each team's 3rd WR, -x for its 2nd TE
(depth order = pre-game long-run snap share), faded exp(-games/6). Poisson GLM, offset = model projection."""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
dl, M = sys.argv[1], sys.argv[2]
def cnt(s, p):
    if not isinstance(s, str): return np.nan
    for part in s.split(','):
        k, q = part.strip().split(' ', 1)
        if q == p: return int(k)
    return 0
R = []
for y in (2023, 2024, 2025):
    q = pd.read_parquet(f'{dl}/part_{y}.parquet', columns=['possession_team', 'offense_personnel']); q['season'] = y
    q['is11'] = [(cnt(s, 'RB') == 1) & (cnt(s, 'TE') == 1) & (cnt(s, 'WR') == 3) if isinstance(s, str) else np.nan for s in q.offense_personnel]
    R.append(q.dropna(subset=['offense_personnel']).groupby(['season', 'possession_team']).is11.mean())
R11 = pd.concat(R)
C = pd.read_csv(f'{M}/data/play_callers.csv'); cur = C.set_index(['season', 'team']).caller
rows = []
for (s, t), caller in cur.items():
    if s not in (2024, 2025) or (s - 1, t) not in cur.index or cur[(s - 1, t)] == caller: continue
    prev = [tt for (ss, tt), cc in cur.items() if ss == s - 1 and cc == caller]
    if prev and (s - 1, prev[0]) in R11.index and (s - 1, t) in R11.index:
        rows.append(dict(season=s, posteam=t, d11=R11[(s - 1, prev[0])] - R11[(s - 1, t)]))
D = pd.DataFrame(rows); print(D.round(3).to_string(index=False))
B = pd.read_parquet(f'{M}/data/backtest_predictions.parquet')
B = B[B.position.isin(['WR', 'TE']) & (B.season_type == 'REG')].copy()
B['rk'] = B.groupby(['game_id', 'posteam', 'position']).snap_l.rank(ascending=False, method='first')
B = B.merge(D, on=['season', 'posteam'], how='left'); B['d11'] = B.d11.fillna(0)
role = np.where((B.position == 'WR') & (B.rk == 3), 1.0, np.where((B.position == 'TE') & (B.rk == 2), -1.0, 0.0))
B['x'] = B.d11 * role * np.exp(-B.games_season / 6) * 10
d = B[(B.targets_hat > 0.2) & (B.games_prior >= 3)]
m = d.x != 0
print(f'{m.sum()} depth-player games on mover-caller teams')
for stat in ['targets', 'rec', 'rec_yds']:
    g = sm.GLM(d[stat], d[['x']], family=sm.families.Poisson(), offset=np.log(d[f'{stat}_hat'])).fit(cov_type='HC1')
    p1 = d[f'{stat}_hat'] * np.exp(d.x * g.params.x)
    print(f'  {stat}: coef per +10 pts of 11 personnel {g.params.x:+.3f} (p {g.pvalues.x:.3f}) | MAE on those games {(d[stat] - d[f"{stat}_hat"])[m].abs().mean():.3f} -> {(d[stat] - p1)[m].abs().mean():.3f}')
