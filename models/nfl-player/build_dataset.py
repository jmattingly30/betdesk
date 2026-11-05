"""nflverse play-by-play + snap counts -> one row per offensive skill player per game (2016 on), with the
team's volume, the game's betting line and the per-play yardage pools used by the simulator.

python3 build_dataset.py <dl_dir> <model_dir>
<dl_dir> holds pbp_YYYY.parquet, snaps_YYYY.parquet and players.parquet from github.com/nflverse/nflverse-data releases."""
import sys, glob, os, numpy as np, pandas as pd, pyarrow.parquet as pq

dl, out = sys.argv[1], sys.argv[2]
os.makedirs(f'{out}/data', exist_ok=True)
COLS = ['game_id', 'season', 'season_type', 'week', 'game_date', 'home_team', 'away_team', 'posteam', 'defteam', 'play_type',
        'pass_attempt', 'rush_attempt', 'sack', 'complete_pass', 'interception', 'qb_dropback', 'qb_scramble', 'two_point_attempt',
        'passer_player_id', 'passer_player_name', 'receiver_player_id', 'receiver_player_name', 'rusher_player_id', 'rusher_player_name',
        'passing_yards', 'receiving_yards', 'rushing_yards', 'pass_touchdown', 'rush_touchdown', 'air_yards', 'yardline_100',
        'spread_line', 'total_line', 'home_score', 'away_score', 'roof', 'wind', 'temp']

frames = []
for f in sorted(glob.glob(f'{dl}/pbp_*.parquet')):
    frames.append(pq.read_table(f, columns=COLS).to_pandas())
p = pd.concat(frames, ignore_index=True)
p = p[p.play_type.isin(['pass', 'run', 'qb_kneel']) & (p.two_point_attempt != 1) & p.posteam.notna()].copy()
for c in ['passing_yards', 'receiving_yards', 'rushing_yards']:
    p[c] = p[c].fillna(0)

att = p[(p.pass_attempt == 1) & (p.sack != 1) & p.passer_player_id.notna()]
run = p[(p.rush_attempt == 1) & p.rusher_player_id.notna()]
tgt = att[att.receiver_player_id.notna()]
key = ['game_id', 'posteam']

g_pass = att.groupby(key + ['passer_player_id']).agg(pass_att=('pass_attempt', 'size'), cmp=('complete_pass', 'sum'),
          pass_yds=('passing_yards', 'sum'), pass_td=('pass_touchdown', 'sum'), ints=('interception', 'sum'))
g_pass.index.names = key + ['pid']
g_rush = run.groupby(key + ['rusher_player_id']).agg(carries=('rush_attempt', 'size'), rush_yds=('rushing_yards', 'sum'),
          rush_td=('rush_touchdown', 'sum'), rz_carries=('yardline_100', lambda s: (s <= 10).sum()),
          scrambles=('qb_scramble', lambda s: s.fillna(0).sum()))   # audit F4: QB scrambles vs designed runs
g_rush.index.names = key + ['pid']
g_rec = tgt.groupby(key + ['receiver_player_id']).agg(targets=('pass_attempt', 'size'), rec=('complete_pass', 'sum'),
          rec_yds=('receiving_yards', 'sum'), rec_td=('pass_touchdown', 'sum'), air_yds=('air_yards', 'sum'),
          rz_targets=('yardline_100', lambda s: (s <= 10).sum()))
g_rec.index.names = key + ['pid']
pg = g_pass.join(g_rush, how='outer').join(g_rec, how='outer').fillna(0).reset_index()

team = p.groupby(key).agg(team_dropbacks=('qb_dropback', 'sum'), team_pass_att=('pass_attempt', lambda s: 0),
                          team_carries=('rush_attempt', 'sum'))
team['team_pass_att'] = att.groupby(key).size()
team['team_targets'] = tgt.groupby(key).size()
team['team_air'] = tgt.groupby(key).air_yards.sum()
team['team_rz_carries'] = run[run.yardline_100 <= 10].groupby(key).size()
team['team_rz_targets'] = tgt[tgt.yardline_100 <= 10].groupby(key).size()
team = team.fillna(0).reset_index()

games = p.groupby('game_id').agg(season=('season', 'first'), season_type=('season_type', 'first'), week=('week', 'first'),
        game_date=('game_date', 'first'), home_team=('home_team', 'first'), away_team=('away_team', 'first'),
        spread_line=('spread_line', 'first'), total_line=('total_line', 'first'), home_score=('home_score', 'first'),
        away_score=('away_score', 'first'), roof=('roof', 'first'), wind=('wind', 'first'), temp=('temp', 'first')).reset_index()

# snap counts (pfr ids) -> gsis ids; defines who played and their snap share
pl = pd.read_parquet(f'{dl}/players.parquet', columns=['gsis_id', 'pfr_id', 'display_name', 'position', 'birth_date', 'rookie_season', 'draft_pick'])
sn = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{dl}/snaps_*.parquet'))], ignore_index=True)
sn = sn[sn.offense_snaps > 0]
sn = sn.merge(pl[['gsis_id', 'pfr_id']].dropna(), left_on='pfr_player_id', right_on='pfr_id', how='left')
sn = sn.rename(columns={'gsis_id': 'pid', 'team': 'posteam', 'position': 'snap_pos'})
# snap ids use pfr team codes in a few seasons; align to pbp codes through the game's two teams
fix = {'LVR': 'LV', 'OAK': 'LV', 'SDG': 'LAC', 'SD': 'LAC', 'STL': 'LA', 'LAR': 'LA', 'GNB': 'GB', 'KAN': 'KC', 'NOR': 'NO', 'NWE': 'NE', 'SFO': 'SF', 'TAM': 'TB'}
sn['posteam'] = sn.posteam.replace(fix)
sn = sn[['game_id', 'posteam', 'pid', 'player', 'snap_pos', 'offense_snaps', 'offense_pct']]
ok_team = set(map(tuple, team[key].values))
bad = ~sn[key].apply(tuple, axis=1).isin(ok_team)
print('snap rows with unmatched team code', bad.sum(), sn[bad].posteam.unique()[:20])
sn_id = sn[sn.pid.notna()].drop_duplicates(key + ['pid'])
# a few games (e.g. one Super Bowl) list players under the wrong team: trust the play-by-play team when the player touched the ball
pbp_team = pg.drop_duplicates(['game_id', 'pid']).set_index(['game_id', 'pid']).posteam
fixed = pd.Series(list(zip(sn_id.game_id, sn_id.pid))).map(pbp_team).to_numpy()
sn_id = sn_id.assign(posteam=np.where(pd.isna(fixed), sn_id.posteam, fixed))

df = sn_id.merge(pg, on=key + ['pid'], how='outer')
print('touch rows without snaps', df.offense_snaps.isna().sum(), 'of', len(df))
df = df.merge(team, on=key, how='inner').merge(games, on='game_id', how='inner')
df['home'] = (df.posteam == df.home_team).astype(int)
df['opp'] = np.where(df.home == 1, df.away_team, df.home_team)
# spread_line = expected home margin (positive = home favoured); team view
df['spread'] = np.where(df.home == 1, df.spread_line, -df.spread_line)
df['implied'] = df.total_line / 2 + df.spread / 2
df['team_score'] = np.where(df.home == 1, df.home_score, df.away_score)
team_snaps = sn.groupby(key).offense_snaps.max().rename('team_snaps')
df = df.merge(team_snaps, on=key, how='left')

meta = pl.drop_duplicates('gsis_id').set_index('gsis_id')
df['position'] = df.pid.map(meta.position)
df['position'] = df.position.where(df.position.isin(['QB', 'RB', 'WR', 'TE', 'FB']), df.snap_pos)
df.loc[df.position == 'FB', 'position'] = 'RB'
df['name'] = df.pid.map(meta.display_name).fillna(df.player)
df['rookie_season'] = df.pid.map(meta.rookie_season)
stat = ['pass_att', 'cmp', 'pass_yds', 'pass_td', 'ints', 'carries', 'rush_yds', 'rush_td', 'rz_carries', 'targets', 'rec', 'rec_yds',
        'rec_td', 'air_yds', 'rz_targets', 'scrambles']
df[stat] = df[stat].fillna(0)
df['offense_snaps'] = df.offense_snaps.fillna(0)
df['offense_pct'] = df.offense_pct.fillna(df.offense_snaps / df.team_snaps).fillna(0)
df = df[df.position.isin(['QB', 'RB', 'WR', 'TE'])]
df['game_date'] = pd.to_datetime(df.game_date)
df = df.drop(columns=['player', 'snap_pos']).sort_values(['game_date', 'game_id', 'posteam', 'pid']).reset_index(drop=True)
df.to_parquet(f'{out}/data/player_games.parquet', index=False)
print(df.shape, df.season.min(), df.season.max(), df.position.value_counts().to_dict())

# per-play yardage pools for the simulator (regular + post season, all years)
pools = pd.concat([
    att[att.complete_pass == 1].assign(kind='catch', pos=att.receiver_player_id)[['kind', 'pos', 'receiving_yards', 'season']].rename(columns={'receiving_yards': 'yds'}),
    run.assign(kind='carry', pos=run.rusher_player_id)[['kind', 'pos', 'rushing_yards', 'season']].rename(columns={'rushing_yards': 'yds'})])
pools['pos'] = pools.pos.map(meta.position).replace({'FB': 'RB'})
pools = pools[pools.pos.isin(['QB', 'RB', 'WR', 'TE'])]
pools.to_parquet(f'{out}/data/play_yards.parquet', index=False)
print(pools.groupby(['kind', 'pos']).yds.agg(['size', 'mean', 'median']))
