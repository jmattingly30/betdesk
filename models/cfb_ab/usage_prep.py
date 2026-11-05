"""Player usage per team-game from ESPN CFB play-by-play (sportsdataverse), for the missing-star Model A input.
python3 usage_prep.py <pbp dir with espn_YYYY.parquet> [years...] -> data/usage_games.parquet
rows: season, week, game_id, team, player, rush_yds, rec_yds, pass_att, pass_yds, touches (game order kept by week/date)."""
import sys, glob, os, pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); PB = sys.argv[1]; YRS = [int(y) for y in sys.argv[2:]] or range(2021, 2027)
C = ['season', 'week', 'game_id', 'pos_team_id', 'rusher_player_name', 'yds_rushed', 'passer_player_name', 'receiver_player_name', 'yds_receiving',
     'rush', 'pass', 'pass_attempt', 'completion', 'statYardage', 'sack', 'game_date', 'seasonType']
out = []
for y in YRS:
    f = f"{PB}/espn_{y}.parquet"
    if not os.path.exists(f): continue
    have = set(__import__('pyarrow.parquet', fromlist=['x']).read_schema(f).names); d = pd.read_parquet(f, columns=[c for c in C if c in have])
    d = d[d.pos_team_id.notna()]
    r = d[(d['rush'] == True) & d.rusher_player_name.notna()].groupby(['season', 'week', 'game_id', 'pos_team_id', 'rusher_player_name']).agg(rush_yds=('yds_rushed', 'sum'), rush_n=('yds_rushed', 'size'))
    c = d[(d['pass'] == True) & d.receiver_player_name.notna()].groupby(['season', 'week', 'game_id', 'pos_team_id', 'receiver_player_name']).agg(rec_yds=('yds_receiving', 'sum'), tgt=('yds_receiving', 'size'))
    p = d[(d['pass_attempt'] == True) & d.passer_player_name.notna()].groupby(['season', 'week', 'game_id', 'pos_team_id', 'passer_player_name']).agg(pass_att=('pass_attempt', 'size'), pass_yds=('yds_receiving', 'sum'))
    for X in (r, c, p): X.index = X.index.set_names(['season', 'week', 'game_id', 'team', 'player'])
    u = pd.concat([r, c, p], axis=1).fillna(0).reset_index()
    dt = d.groupby('game_id').game_date.first() if 'game_date' in d else None
    u['date'] = u.game_id.map(dt) if dt is not None else None
    st = d.groupby('game_id').seasonType.first(); u['stype'] = u.game_id.map(st)
    out.append(u); print(y, len(u), flush=True)
U = pd.concat(out, ignore_index=True); U['team'] = U.team.astype(int); U['game_id'] = U.game_id.astype(str)
F = f"{HERE}/data/usage_games.parquet"
if os.path.exists(F):   # replace only the seasons pulled this run (live refresh pulls 2026 alone)
    old = pd.read_parquet(F); U = pd.concat([old[~old.season.isin(U.season.unique())], U], ignore_index=True)
U.to_parquet(F); print(U.shape)
