"""Option 1 test (2026-10-10): play-caller priors for position target shares.

python3 caller_priors_test.py <dl_dir> <model_dir>
When a team's play-caller changed, the model's share history still reflects the OLD caller. Shift each WR/TE/RB's
projection by how the new caller split targets by position on his previous team (league average for a first-time
caller) vs how this team split them last season, fading over the new season:
  x = log(caller's previous WR|TE|RB target share / team's last-season share for that position) x exp(-games into season / 6)
Poisson GLM on the out-of-sample projections (backtest_predictions.parquet, 2024-25), offset = projection.
Coefficient 1 = apply the caller's split fully. Also: does it help only new-caller teams (it is 0 elsewhere)?"""
import sys, numpy as np, pandas as pd, statsmodels.api as sm
dl, M = sys.argv[1], sys.argv[2]
pos = pd.read_parquet(f'{dl}/players.parquet', columns=['gsis_id', 'position']).set_index('gsis_id').position.replace({'FB': 'RB'})
P = pd.concat([pd.read_parquet(f'{dl}/pbp_{y}.parquet', columns=['season', 'season_type', 'posteam', 'receiver_player_id']) for y in (2023, 2024, 2025)])
P = P[(P.season_type == 'REG') & P.receiver_player_id.notna()]; P['tpos'] = P.receiver_player_id.map(pos)
SH = P[P.tpos.isin(['WR', 'TE', 'RB'])].groupby(['season', 'posteam']).tpos.value_counts(normalize=True).unstack().fillna(0)
LG = SH.groupby(level=0).mean()
C = pd.read_csv(f'{M}/data/play_callers.csv')
cur = C.set_index(['season', 'team']).caller
rows = []
for (s, t), caller in cur.items():
    if s not in (2024, 2025) or (s - 1, t) not in cur.index: continue
    if cur[(s - 1, t)] == caller: continue
    prev = [tt for (ss, tt), cc in cur.items() if ss == s - 1 and cc == caller]
    prior = SH.loc[(s - 1, prev[0])] if prev else LG.loc[s - 1]
    team_last = SH.loc[(s - 1, t)]
    for p_ in ['WR', 'TE', 'RB']:
        rows.append(dict(season=s, posteam=t, position=p_, lr=np.log(prior[p_] / team_last[p_]), caller=caller, had_prior=bool(prev)))
R = pd.DataFrame(rows)
print('new-caller teams:', R.drop_duplicates(['season', 'posteam']).shape[0], '| with previous play-calling job:', R.drop_duplicates(['season', 'posteam']).had_prior.sum())
B = pd.read_parquet(f'{M}/data/backtest_predictions.parquet')
B = B[(B.snap_l >= 0.25) & (B.games_prior >= 3) & B.position.isin(['WR', 'TE', 'RB']) & (B.season_type == 'REG')].copy()
B = B.merge(R[['season', 'posteam', 'position', 'lr', 'had_prior']], on=['season', 'posteam', 'position'], how='left')
B['new_caller'] = B.lr.notna(); B['lr'] = B.lr.fillna(0)
for fade in (4, 6, 10, 99):
    B[f'x{fade}'] = B.lr * np.exp(-B.games_season / fade)
for stat in ['targets', 'rec', 'rec_yds']:
    d = B[B[f'{stat}_hat'] > 0.2]
    print(f'\n{stat}: {d.new_caller.sum()} player-games on new-caller teams (of {len(d)})')
    for fade in (4, 6, 10, 99):
        x = d[[f'x{fade}']]
        g = sm.GLM(d[stat], x, family=sm.families.Poisson(), offset=np.log(d[f'{stat}_hat'])).fit(cov_type='cluster', cov_kwds={'groups': d.posteam + d.season.astype(str)})
        tr, te = d.season == 2024, d.season == 2025
        gt = sm.GLM(d[stat][tr], x[tr], family=sm.families.Poisson(), offset=np.log(d[f'{stat}_hat'][tr])).fit()
        nc = te & d.new_caller; early = nc & (d.games_season < 6)
        p1 = d[f'{stat}_hat'] * np.exp(x.iloc[:, 0] * gt.params.iloc[0])
        def mae(m, p): return (d[stat][m] - p[m]).abs().mean()
        print(f'  fade {fade:>2}: coef {g.params.iloc[0]:+.2f} (p {g.pvalues.iloc[0]:.3f}, fit 2024 {gt.params.iloc[0]:+.2f}) | 2025 new-caller MAE {mae(nc, d[f"{stat}_hat"]):.3f} -> {mae(nc, p1):.3f}'
              f' | first 6 games {mae(early, d[f"{stat}_hat"]):.3f} -> {mae(early, p1):.3f}')

# only callers with a previous play-calling job (the movers: their split is real information, not a league average)
d = B[(B.targets_hat > 0.2)]
for fade in (6, 99):
    m = d.had_prior.fillna(False).astype(bool)
    x = (d[f'x{fade}'] * m).to_frame('x')
    g = sm.GLM(d.targets, x, family=sm.families.Poisson(), offset=np.log(d.targets_hat)).fit(cov_type='cluster', cov_kwds={'groups': d.posteam + d.season.astype(str)})
    p1 = d.targets_hat * np.exp(x.x * g.params.iloc[0]); e = m & (d.games_season < 6)
    print(f'movers only, targets, fade {fade}: {m.sum()} player-games, coef {g.params.iloc[0]:+.2f} (p {g.pvalues.iloc[0]:.3f}) | in-sample MAE movers {(d.targets - d.targets_hat)[m].abs().mean():.3f} -> {(d.targets - p1)[m].abs().mean():.3f}, first 6 games {(d.targets - d.targets_hat)[e].abs().mean():.3f} -> {(d.targets - p1)[e].abs().mean():.3f}')
