"""2026-27 preseason roster links from ESPN's official 2027 rosters (audit fix C6 for live use).

Historical seasons build roster continuity from who played in a team's first five games (prep_returning.py), which the
prior then 'knows' early in those seasons. For the coming season the real preseason roster exists, so the prior for 2027
uses it: for each player on a 2027 roster, the team he played for in 2025-26 and his share of that team's minutes.
Appends season-2027 rows to data/cbb_roster_links.parquet (same columns).
usage: python3 prep_roster_2027.py <hoopR dir> data"""
import sys, pandas as pd
d, out = sys.argv[1:3]
R = pd.read_parquet(f'{d}/rosters_2027.parquet', columns=['team_id', 'athlete_id'])
R['team_id'] = pd.to_numeric(R.team_id); R['athlete_id'] = pd.to_numeric(R.athlete_id)
# ESPN's 2027 roster file is a stale copy of the 2026 roster until it is updated for the new season (checked 2026-10-09:
# 97% of team-player pairs identical, classes not advanced). Refuse to use it until it changes.
R26 = pd.read_parquet(f'{d}/rosters_2026.parquet', columns=['team_id', 'athlete_id'])
same = len(set(zip(R.team_id, R.athlete_id)) & set(zip(pd.to_numeric(R26.team_id), pd.to_numeric(R26.athlete_id)))) / len(R)
if same > 0.7:
    raise SystemExit(f'2027 rosters are still {same:.0%} identical to 2026 (not updated for transfers/graduates yet); rerun later.')
b = pd.read_parquet(f'{d}/player_box_2026.parquet', columns=['athlete_id', 'team_id', 'minutes'])
b['minutes'] = pd.to_numeric(b.minutes, errors='coerce').fillna(0); b = b.dropna(subset=['athlete_id'])
b['athlete_id'] = pd.to_numeric(b.athlete_id); b['team_id'] = pd.to_numeric(b.team_id)
tot = b.groupby('team_id').minutes.sum()
pm = b.groupby(['athlete_id', 'team_id']).minutes.sum().reset_index(); pm['share'] = pm.minutes / pm.team_id.map(tot)
l = R.merge(pm.rename(columns={'team_id': 'prev_team'}), on='athlete_id', how='left').rename(columns={'team_id': 'team'})
l['season'] = 2027; l['returning'] = l.team == l.prev_team
L = pd.read_parquet(f'{out}/cbb_roster_links.parquet')
L = pd.concat([L[L.season != 2027], l[L.columns.intersection(l.columns)]], ignore_index=True)
L.to_parquet(f'{out}/cbb_roster_links.parquet')
r = l.groupby('team').apply(lambda x: pd.Series({'ret_min': x.share[x.returning].sum(), 'xfer_min': x.share[~x.returning].fillna(0).sum()}))
print('2027 rosters: teams %d players %d | mean returning minutes %.3f, transfer minutes %.3f' % (r.shape[0], len(l), r.ret_min.mean(), r.xfer_min.mean()))
