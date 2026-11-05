"""Aggregate nflverse play-by-play (2006-2025) to one row per team per game: offense and defense
efficiency, luck-prone stats (turnovers, fumble recoveries, close games), special teams, and the result."""
import sys, glob, numpy as np, pandas as pd, pyarrow.parquet as pq
pbp_dir, out = sys.argv[1], sys.argv[2]
COLS = ['game_id','season','season_type','week','game_date','home_team','away_team','posteam','defteam','play_type','pass','rush',
        'epa','success','wp','down','interception','fumble','fumble_lost','sack','penalty','third_down_converted','third_down_failed',
        'yards_gained','special_teams_play','qb_kneel','qb_spike','home_score','away_score','drive','fixed_drive_result',
        'drive_start_yard_line','yardline_100','spread_line','total_line','cpoe','pass_oe','air_yards','qb_dropback']
frames = []
for f in sorted(glob.glob(f'{pbp_dir}/pbp_*.parquet')):
    s = set(pq.read_schema(f).names)
    p = pq.read_table(f, columns=[c for c in COLS if c in s]).to_pandas()
    for c in COLS:
        if c not in p: p[c] = np.nan
    frames.append(p)
p = pd.concat(frames, ignore_index=True)
p = p[p.season_type == 'REG']
sc = p[(p['pass'] == 1) | (p['rush'] == 1)].copy()
sc = sc[sc.epa.notna() & (sc.qb_kneel != 1) & (sc.qb_spike != 1)]
sc['nongarb'] = sc.wp.between(0.1, 0.9)
sc['early'] = sc.down.isin([1, 2])
sc['explosive'] = ((sc['pass'] == 1) & (sc.yards_gained >= 20)) | ((sc['rush'] == 1) & (sc.yards_gained >= 10))
g = sc.groupby(['game_id','posteam'])
def agg(mask=None, pre=''):
    s = sc if mask is None else sc[mask]
    gg = s.groupby(['game_id','posteam'])
    return pd.DataFrame({pre+'plays': gg.size(), pre+'epa': gg.epa.mean(), pre+'sr': gg.success.mean()})
off = agg()
off = off.join(agg(sc['pass'] == 1, 'pass_')).join(agg(sc['rush'] == 1, 'rush_')).join(agg(sc.early, 'early_'))
off = off.join(agg(sc.nongarb, 'ng_'))
off['explosive'] = g.explosive.mean()
off['sack_rate'] = sc[sc.qb_dropback == 1].groupby(['game_id','posteam']).sack.mean()
off['pass_rate_oe'] = g.pass_oe.mean()
off['cpoe'] = g.cpoe.mean()
off['third_conv'] = g.third_down_converted.sum() / (g.third_down_converted.sum() + g.third_down_failed.sum())
# turnovers and fumble luck (all plays incl. special teams)
allp = p[p.posteam.notna()]
ga = allp.groupby(['game_id','posteam'])
off['ints'] = ga.interception.sum(); off['fumbles'] = ga.fumble.sum(); off['fumbles_lost'] = ga.fumble_lost.sum()
# drives: start field position, points per drive, red zone TD rate
dr = allp.dropna(subset=['drive']).groupby(['game_id','posteam','drive']).agg(res=('fixed_drive_result','first'), y100=('yardline_100','first'), minyl=('yardline_100','min'))
dr['td'] = dr.res.eq('Touchdown'); dr['rz'] = dr.minyl <= 20
dg = dr.groupby(['game_id','posteam'])
off['drives'] = dg.size(); off['start_y100'] = dg.y100.mean()
off['rz_td_rate'] = dr[dr.rz].groupby(['game_id','posteam']).td.mean()
st = p[p.special_teams_play == 1].groupby(['game_id','posteam']).epa.sum()
off['st_epa'] = st
off = off.reset_index().rename(columns={'posteam':'team'})
games = p.groupby('game_id').agg(season=('season','first'), week=('week','first'), date=('game_date','first'), home=('home_team','first'),
        away=('away_team','first'), hs=('home_score','max'), as_=('away_score','max'), spread_line=('spread_line','first'), total_line=('total_line','first')).reset_index()
t = off.merge(games, on='game_id')
t['is_home'] = (t.team == t.home).astype(int)
t['opp'] = np.where(t.is_home == 1, t.away, t.home)
t['pf'] = np.where(t.is_home == 1, t.hs, t.as_); t['pa'] = np.where(t.is_home == 1, t.as_, t.hs)
d = t.drop(columns=['season','week','date','home','away','hs','as_','spread_line','total_line','is_home','opp','pf','pa']).add_prefix('d_').rename(columns={'d_game_id':'game_id','d_team':'opp'})
t = t.merge(d, on=['game_id','opp'])
t.to_csv(out, index=False)
print(t.shape, t.season.min(), t.season.max())
