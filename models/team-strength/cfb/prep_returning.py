"""Returning offensive production for every CFB team-season (audit fix T1), from ESPN player box scores.
ret_pass  = share of last season's passing yards thrown by players who play for the team again this season
ret_skill = same for rushing + receiving yards;  ret_off = 0.5 ret_pass + 0.5 ret_skill
'Plays again' = appears in one of the team's first three games this season. ESPN rosters are not archived, so this
knows who was healthy and eligible in weeks 1-3 (the same small leak as the college basketball prior, audit C6);
game_model.py reports weeks 1-3 separately. Defensive box stats exist only from 2023, so there is no defensive term.
usage: python3 prep_returning.py <dir with pb_YYYY.parquet> <cfb_games.parquet> <out.csv>"""
import sys, numpy as np, pandas as pd
d, games, out = sys.argv[1:4]
g = pd.read_parquet(games)[['game_id', 'season', 'date', 'home_id', 'away_id']]
B = []
for y in range(2005, 2026):
    p = pd.read_parquet(f'{d}/pb_{y}.parquet', columns=['category', 'athlete_id', 'team_id', 'game_id', 'season', 'passingYards', 'rushingYards', 'receivingYards'])
    B.append(p)
B = pd.concat(B, ignore_index=True)
for c in ['passingYards', 'rushingYards', 'receivingYards', 'athlete_id', 'team_id', 'game_id']: B[c] = pd.to_numeric(B[c], errors='coerce')
B = B.dropna(subset=['athlete_id', 'team_id', 'game_id'])
B['pass_y'] = B.passingYards.fillna(0).clip(lower=0); B['skill_y'] = (B.rushingYards.fillna(0) + B.receivingYards.fillna(0)).clip(lower=0)
tot = B.groupby(['season', 'team_id', 'athlete_id'])[['pass_y', 'skill_y']].sum().reset_index()
# each team's first three games of each season
tg = pd.concat([g[['game_id', 'season', 'date', 'home_id']].rename(columns={'home_id': 'team_id'}), g[['game_id', 'season', 'date', 'away_id']].rename(columns={'away_id': 'team_id'})])
tg = tg.sort_values('date'); tg['n'] = tg.groupby(['season', 'team_id']).cumcount()
first3 = tg[tg.n < 3][['game_id', 'team_id']]
early = B.merge(first3, on=['game_id', 'team_id'])[['season', 'team_id', 'athlete_id']].drop_duplicates()
rows = []
for s in range(2006, 2026):
    last = tot[tot.season == s - 1]; now = set(zip(early[early.season == s].team_id, early[early.season == s].athlete_id))
    if last.empty: continue
    last = last.assign(back=[(t, a) in now for t, a in zip(last.team_id, last.athlete_id)])
    agg = last.groupby('team_id').apply(lambda x: pd.Series({'ret_pass': (x.pass_y * x.back).sum() / max(x.pass_y.sum(), 1), 'ret_skill': (x.skill_y * x.back).sum() / max(x.skill_y.sum(), 1),
                                                              'pass_tot': x.pass_y.sum()}), include_groups=False).reset_index()
    agg['season'] = s; rows.append(agg)
R = pd.concat(rows, ignore_index=True)
R['ret_off'] = 0.5 * R.ret_pass + 0.5 * R.ret_skill
R.to_csv(out, index=False)
print(R.groupby('season')[['ret_pass', 'ret_skill', 'ret_off']].mean().round(3).to_string()); print(len(R), 'team-seasons')
