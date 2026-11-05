"""ESPN player box scores (sportsdataverse) -> one row per player-game, plus market lines from team-strength.
usage: python3 build_dataset.py <download_dir> <out_dir>
download_dir holds player_box_YYYY.parquet (github.com/sportsdataverse/sportsdataverse-data releases/espn_nba_player_boxscores)."""
import sys, glob, numpy as np, pandas as pd
d, out = sys.argv[1], sys.argv[2]
cols = {'field_goals_made':'fgm','field_goals_attempted':'fga','three_point_field_goals_made':'fg3m','three_point_field_goals_attempted':'fg3a',
        'free_throws_made':'ftm','free_throws_attempted':'fta','offensive_rebounds':'oreb','defensive_rebounds':'dreb','rebounds':'reb',
        'assists':'ast','steals':'stl','blocks':'blk','turnovers':'tov','fouls':'pf','points':'pts','minutes':'min'}
b = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{d}/pb_*.parquet'))], ignore_index=True)
b = b[b.season_type.isin([2, 3])].rename(columns=cols)
b = b.rename(columns={'athlete_id':'pid','athlete_display_name':'name','team_abbreviation':'team','opponent_team_abbreviation':'opp',
                      'athlete_position_abbreviation':'pos'})
b = b[b.pid.notna()].copy(); b['game_id'] = b.game_id.astype(int); b['pid'] = b.pid.astype(int)
b['date'] = pd.to_datetime(b.game_date)
b['played'] = (b['min'].fillna(0) > 0)
b['pos_grp'] = b.pos.map({'PG':'G','SG':'G','G':'G','SF':'F','PF':'F','F':'F','C':'C'}).fillna('F')
b['playoff'] = (b.season_type == 3).astype(int)
b['is_home'] = (b.home_away == 'home').astype(int)
b['dnp_injury'] = b.did_not_play.astype(bool) & ~b.reason.fillna('').str.contains("COACH", case=False)
keep = ['game_id','season','date','playoff','pid','name','team','opp','is_home','pos','pos_grp','starter','played','did_not_play','dnp_injury','reason',
        'team_score','opponent_team_score'] + list(cols.values())
b = b[keep].drop_duplicates(['game_id','pid']).sort_values(['date','game_id','team','pid']).reset_index(drop=True)
for c in cols.values(): b[c] = b[c].astype(float)
# market: spread and total from the team-strength prep (line = expected HOME margin)
g = pd.read_parquet(f'{out}/../team-strength/data/nba_games.parquet')[['game_id','home','line','total_line','home_rest','away_rest']]
b = b.merge(g, on='game_id', how='left')
sgn = np.where(b.is_home == 1, 1, -1)
b['team_spread'] = sgn * b.line                      # expected margin for the player's team
b['team_implied'] = (b.total_line + b.team_spread) / 2
b['rest'] = np.where(b.is_home == 1, b.home_rest, b.away_rest)
b = b.drop(columns=['home','line','home_rest','away_rest'])
b.to_parquet(f'{out}/data/player_games.parquet')
p = b[b.played]
print('rows', b.shape, 'played', len(p), 'seasons', p.groupby('season').size().to_dict())
print('market coverage by season', p.groupby('season').total_line.apply(lambda s: round(s.notna().mean(), 3)).to_dict())
