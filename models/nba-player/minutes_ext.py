"""NBA-2 minutes upgrade (2026-10-11): pre-game minutes features the original model lacked.

  g_back     played games since returning from a 3+ team-game absence (0 = first game back)
  miss_prev  length of that absence in team games (0 if the latest stretch did not follow a 3+ game absence)
  ramp       1 / (1 + g_back) after a 3+ game absence, else 0 (minutes restrictions fade over a few games)
  m3         mean minutes of the last 3 games played;  m_trend = m3 minus the last-15 mean
  m_sd10     standard deviation of the last 10 games' minutes (rotation stability)
  rot_depth  players available tonight for this team who average 15+ recent minutes
Out of sample 2024-25 + 2025-26 (backtest.py): minutes MAE 4.80 -> 4.61 (-3.9%, t 26), PRA -1.7%, points -1.1%;
first 3 games back from an absence: minutes -6.9%, PRA -4.1%. add(df) works on history + upcoming rows (future=True)."""
import numpy as np, pandas as pd
NEW = ['g_back', 'miss_prev', 'ramp', 'm3', 'm_sd10', 'm_trend', 'rot_depth']

def add(f):
    f = f.drop(columns=[c for c in NEW if c in f]).copy()
    f['_o'] = np.arange(len(f))
    fut = f['future'].fillna(False).astype(bool) if 'future' in f else pd.Series(False, index=f.index)
    tgm = f[['game_id', 'team', 'season', 'date']].drop_duplicates(['game_id', 'team']).sort_values(['date', 'game_id'])
    tgm['tidx'] = tgm.groupby(['team', 'season']).cumcount()
    f = f.merge(tgm[['game_id', 'team', 'tidx']], on=['game_id', 'team'], how='left')
    fut = f['future'].fillna(False).astype(bool) if 'future' in f else pd.Series(False, index=f.index)
    p = f[f.played.astype(bool) & ((f['min'] > 0) | fut)].sort_values(['pid', 'date', 'game_id']).copy()
    same = (p.pid == p.pid.shift()) & (p.team == p.team.shift()) & (p.season == p.season.shift())
    p['miss_before'] = np.where(same, p.tidx - p.tidx.shift() - 1, np.nan)
    p['grp'] = ((p.miss_before.fillna(0) >= 3) | ~same).groupby(p.pid).cumsum()
    p['g_back'] = p.groupby(['pid', 'grp']).cumcount()
    p['miss_prev'] = p.groupby(['pid', 'grp']).miss_before.transform('first').fillna(0).clip(upper=40)
    p.loc[p.miss_prev < 3, 'miss_prev'] = 0
    g = p.groupby('pid')['min']
    p['m3'] = g.transform(lambda s: s.shift().rolling(3, min_periods=1).mean())
    p['m_sd10'] = g.transform(lambda s: s.shift().rolling(10, min_periods=3).std())
    p['m_trend'] = p.m3 - g.transform(lambda s: s.shift().rolling(15, min_periods=3).mean())
    f = f.merge(p[['_o', 'g_back', 'miss_prev', 'm3', 'm_sd10', 'm_trend']], on='_o', how='left')
    f['rot_depth'] = f.assign(r=(f.played.astype(bool) & (f.ms >= 15)).astype(int)).groupby(['game_id', 'team']).r.transform('sum')
    f['ramp'] = np.where(f.miss_prev >= 3, 1 / (1 + f.g_back.fillna(0)), 0.0)
    return f.sort_values('_o').drop(columns=['_o', 'tidx']).reset_index(drop=True)
