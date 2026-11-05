"""CBB-2 (2026-10-10, Edge research thread): player-built preseason inputs, transfers included.
data/cbb_player_seasons.csv: one row per (athlete, season): minutes per game, games, and value = per-minute Game Score
shrunk toward replacement level (150 minutes of prior), the same box-score value prep_availability.py uses.
priors.py turns tonight's roster (first 5 games, cbb_roster_links.parquet) into team inputs:
  r_val     sum over the roster of last-season minutes per game x (value - replacement), scaled to 200 team minutes
  r_val_lvl the same, each player's value times his last-season team's net rating / 10 (a 20-point player at Duke vs at
            a low-major: the prior learns how much level matters)
  xfer_val  the part of r_val brought in by transfers
usage: python3 prep_player_prior.py <hoopR dir with player_box_YYYY.parquet> data"""
import sys, numpy as np, pandas as pd
d, out = sys.argv[1:3]
COLS = ['game_id', 'season', 'athlete_id', 'team_id', 'minutes', 'points', 'field_goals_made', 'field_goals_attempted', 'free_throws_made',
        'free_throws_attempted', 'offensive_rebounds', 'defensive_rebounds', 'steals', 'assists', 'blocks', 'fouls', 'turnovers']
B = pd.concat([pd.read_parquet(f'{d}/player_box_{y}.parquet', columns=COLS) for y in range(2008, 2027)], ignore_index=True).dropna(subset=['athlete_id'])
B[COLS[4:]] = B[COLS[4:]].apply(pd.to_numeric, errors='coerce').fillna(0)
B = B.drop_duplicates(['game_id', 'athlete_id']); B = B[B.minutes > 0]
B['gms'] = (B.points + 0.4 * B.field_goals_made - 0.7 * B.field_goals_attempted - 0.4 * (B.free_throws_attempted - B.free_throws_made)
            + 0.7 * B.offensive_rebounds + 0.3 * B.defensive_rebounds + B.steals + 0.7 * B.assists + 0.7 * B.blocks - 0.4 * B.fouls - B.turnovers)
S = B.groupby(['athlete_id', 'season']).agg(minutes=('minutes', 'sum'), games=('game_id', 'nunique'), gms=('gms', 'sum'), team=('team_id', 'last')).reset_index()
low = S.minutes / S.games < 12
REPL = float(S.gms[low].sum() / S.minutes[low].sum())
S['value'] = (S.gms + 150 * REPL) / (S.minutes + 150)
S['mpg'] = (S.minutes / S.games).clip(upper=36)
S[['athlete_id', 'season', 'team', 'mpg', 'games', 'value']].to_csv(f'{out}/cbb_player_seasons.csv', index=False)
print('replacement per-minute Game Score %.4f; %d player-seasons' % (REPL, len(S)))
open(f'{out}/cbb_player_repl.txt', 'w').write(str(REPL))
