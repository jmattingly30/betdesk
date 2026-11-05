"""One row per NBA game (ESPN schedule, 2008-2026 seasons) with closing spread (2017-18 on) and per-team box-score
efficiency (possessions, ratings, four factors). Season = ESPN end year (2026 = 2025-26)."""
import sys, glob, json, numpy as np, pandas as pd
d = sys.argv[1]; out_games, out_tg = sys.argv[2:4]
s = pd.read_parquet(f'{d}/nba_nba_schedule_master.parquet')
s = s[s.season_type.isin([2, 3]) & s.status_type_completed.astype(bool) & s.home_score.notna()].copy()
s['home_score'] = s.home_score.astype(float); s['away_score'] = s.away_score.astype(float)
s = s[(s.home_score > 0) & (s.season.between(2008, 2026))]
s['game_id'] = s.game_id.astype(int)
s['date'] = pd.to_datetime(s.date, utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
s['day'] = s.date.dt.normalize()
s['margin'] = s.home_score - s.away_score
s['home'], s['away'] = s.home_abbreviation, s.away_abbreviation
s['neutral'] = s.neutral_site.astype(bool)
s['playoff'] = (s.season_type == 3).astype(int)
# closing lines: Odds API consensus (2020+) by ESPN id, else the covers-style archive by date+score
c = pd.read_parquet(f'{d}/nba_closing_lines_odds_api.parquet')[['game_id','home_point','over_under']]
c['line_api'] = -c.home_point
s = s.merge(c[['game_id','line_api','over_under']], on='game_id', how='left')
a = pd.DataFrame(json.load(open(f'{d}/nba_games-archive.json')))
a['day'] = pd.to_datetime(a.game_date).dt.normalize(); a['line_arc'] = -a.line.astype(float); a['total_arc'] = pd.to_numeric(a.game_over_under, errors='coerce')
a = a.drop_duplicates(['day','home_team_score','visit_team_score'])
s = s.merge(a[['day','home_team_score','visit_team_score','line_arc','total_arc']], left_on=['day','home_score','away_score'],
            right_on=['day','home_team_score','visit_team_score'], how='left')
s['line'] = s.line_api.fillna(s.line_arc); s['total_line'] = s.over_under.fillna(s.total_arc)
s = s.sort_values(['date','game_id']).reset_index(drop=True)
# rest days and back-to-backs
long = pd.concat([s[['game_id','day','season','home']].rename(columns={'home':'team'}), s[['game_id','day','season','away']].rename(columns={'away':'team'})]).sort_values('day')
long['rest'] = long.groupby(['season','team']).day.diff().dt.days.fillna(7).clip(upper=7)
rest = long.set_index(['game_id','team']).rest
s['home_rest'] = rest.reindex(pd.MultiIndex.from_arrays([s.game_id, s.home])).values
s['away_rest'] = rest.reindex(pd.MultiIndex.from_arrays([s.game_id, s.away])).values
s.to_parquet(out_games)
print('games', s.shape, 'with line', s.line.notna().sum(), s[s.line.notna()].groupby('season').size().to_dict())
m = s.line.notna(); print('close MAE', (s.margin[m] - s.line[m]).abs().mean(), 'corr', np.corrcoef(s.margin[m], s.line[m])[0,1])
# team box -> efficiency
b = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{d}/nbabox/team_box_*.parquet'))], ignore_index=True)
b = b[b.game_id.astype(int).isin(set(s.game_id))].copy(); b['game_id'] = b.game_id.astype(int)
num = ['field_goals_made','field_goals_attempted','three_point_field_goals_made','three_point_field_goals_attempted','free_throws_made','free_throws_attempted',
       'offensive_rebounds','defensive_rebounds','turnovers','total_turnovers','team_score','assists','steals','blocks','fouls']
for c_ in num: b[c_] = pd.to_numeric(b[c_], errors='coerce')
b['tov'] = b.total_turnovers.fillna(b.turnovers)
b['poss'] = b.field_goals_attempted - b.offensive_rebounds + b.tov + 0.44*b.free_throws_attempted
o = b[['game_id','team_abbreviation','opponent_team_abbreviation','team_score','poss','field_goals_made','field_goals_attempted','three_point_field_goals_made',
       'three_point_field_goals_attempted','free_throws_made','free_throws_attempted','offensive_rebounds','defensive_rebounds','tov']].copy()
o.columns = ['game_id','team','opp','pts','poss','fgm','fga','tpm','tpa','ftm','fta','orb','drb','tov']
o = o.drop_duplicates(['game_id','team'])
o = o.merge(o[['game_id','team','pts','poss','fgm','fga','tpm','tpa','ftm','fta','orb','drb','tov']].rename(columns=lambda x: x if x in ('game_id',) else 'o_'+x).rename(columns={'o_team':'opp'}), on=['game_id','opp'])
P = (o.poss + o.o_poss)/2
o['ortg'] = 100*o.pts/P; o['drtg'] = 100*o.o_pts/P; o['net'] = o.ortg - o.drtg; o['pace'] = P
o['efg'] = (o.fgm + 0.5*o.tpm)/o.fga; o['o_efg'] = (o.o_fgm + 0.5*o.o_tpm)/o.o_fga
o['tov_pct'] = o.tov/P; o['o_tov_pct'] = o.o_tov/P
o['orb_pct'] = o.orb/(o.orb + o.o_drb); o['o_orb_pct'] = o.o_orb/(o.o_orb + o.drb)
o['ftr'] = o.fta/o.fga; o['o_ftr'] = o.o_fta/o.o_fga
o['tp_pct'] = o.tpm/o.tpa; o['o_tp_pct'] = o.o_tpm/o.o_tpa; o['tpa_rate'] = o.tpa/o.fga; o['o_tpa_rate'] = o.o_tpa/o.o_fga
o['two_pct'] = (o.fgm - o.tpm)/(o.fga - o.tpa); o['o_two_pct'] = (o.o_fgm - o.o_tpm)/(o.o_fga - o.o_tpa)
o['ft_pct'] = o.ftm/o.fta; o['o_ft_pct'] = o.o_ftm/o.o_fta
o = o.merge(s[['game_id','season','date','home','playoff']], on='game_id')
o['is_home'] = (o.team == o.home).astype(int)
o.to_parquet(out_tg); print('team-games', o.shape)
