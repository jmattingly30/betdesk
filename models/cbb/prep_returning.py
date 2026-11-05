"""Preseason roster continuity from ESPN player box scores.
data/cbb_roster_links.parquet: for each season y, team, and player who appears for the team in its first 5 games of y,
the team(s) he played for in y-1 and his share of that team's y-1 minutes (0 if he didn't play D1 in y-1).
data/cbb_returning.csv: per team-season, ret_min (share of last season's minutes that is back) and xfer_min
(last-season minutes brought in by newcomers, in team-seasons). The first-5-games roster keeps this close to preseason knowledge.
usage: python3 prep_returning.py <hoopR dir> <out dir>"""
import sys, pandas as pd, numpy as np
d, out = sys.argv[1:3]
P = {}
for y in range(2008, 2027):
    b = pd.read_parquet(f'{d}/player_box_{y}.parquet', columns=['game_id', 'game_date', 'athlete_id', 'team_id', 'minutes'])
    b['minutes'] = pd.to_numeric(b.minutes, errors='coerce').fillna(0)
    P[y] = b.dropna(subset=['athlete_id'])
links = []
for y in range(2009, 2027):
    last, cur = P[y - 1], P[y]
    tot = last.groupby('team_id').minutes.sum()
    pm = last.groupby(['athlete_id', 'team_id']).minutes.sum().reset_index()
    pm['share'] = pm.minutes / pm.team_id.map(tot)
    g5 = cur.drop_duplicates(['team_id', 'game_id']).sort_values('game_date').groupby('team_id').head(5)[['team_id', 'game_id']]
    roster = cur[cur.minutes > 0].merge(g5, on=['team_id', 'game_id'])[['team_id', 'athlete_id']].drop_duplicates()
    l = roster.merge(pm.rename(columns={'team_id': 'prev_team'}), on='athlete_id', how='left')
    l['season'] = y; links.append(l)
L = pd.concat(links, ignore_index=True).rename(columns={'team_id': 'team'})
L['returning'] = L.team == L.prev_team
L.to_parquet(f'{out}/cbb_roster_links.parquet')
r = L.groupby(['season', 'team']).apply(lambda x: pd.Series({'ret_min': x.share[x.returning].sum(), 'xfer_min': x.share[~x.returning].fillna(0).sum()})).reset_index()
r.to_csv(f'{out}/cbb_returning.csv', index=False)
print(r.groupby('season')[['ret_min', 'xfer_min']].mean().round(3).to_string())
