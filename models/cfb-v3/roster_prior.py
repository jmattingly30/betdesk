"""CFB-2 (2026-10-10, Edge research thread): roster-built preseason inputs for every team-season, transfers included.

The v3 prior already uses returning production (cfb_returning.csv), but it only counts players who come back to the SAME
team, so a transfer QB counts as zero. This file follows each player to wherever he plays next:
  qb_anya   starting QB's (most attempts in the team's first 3 games) last-season adjusted net yards per attempt,
            (yards + 20 TD - 45 INT) / attempts, from ANY team, shrunk toward the league (150 attempts)
  qb_lvl    that QB's last-season team strength (final Elo - 1500) / 100, so 8.0 ANY/A in the MAC is not 8.0 in the SEC
  qb_new    1 if the starting QB played for another team last season (a transfer), qb_none 1 if he had no passing last season
  avail_skill  last-season rushing + receiving yards (any team) of this season's skill players / league mean, Elo-scaled
  xfer_skill   the part of avail_skill that came from transfers
'This season's roster' = players in the team's first 3 games (ESPN rosters are not archived): the same small leak as
cfb_returning.csv (it knows who was healthy and eligible in weeks 1-3); build.py reports weeks 1-3 separately.
Defensive box stats only exist from 2023, so there is no defensive roster term yet.

usage: python3 roster_prior.py <dir with pb_YYYY.parquet> <cfb_games.parquet> <out.csv>"""
import sys, os, numpy as np, pandas as pd
d, games, out = sys.argv[1:4]
g = pd.read_parquet(games)
B = []
for y in range(2013, 2030):
    f = f'{d}/pb_{y}.parquet'
    if not os.path.exists(f): continue
    p = pd.read_parquet(f, columns=['category', 'athlete_id', 'team_id', 'game_id', 'season', 'completions/passingAttempts', 'passingYards',
                                    'passingTouchdowns', 'interceptions', 'rushingYards', 'receivingYards'])
    B.append(p)
B = pd.concat(B, ignore_index=True)
B['att'] = pd.to_numeric(B['completions/passingAttempts'].astype(str).str.split('/').str[1], errors='coerce')
for c in ['passingYards', 'passingTouchdowns', 'interceptions', 'rushingYards', 'receivingYards', 'athlete_id', 'team_id', 'game_id', 'season']:
    B[c] = pd.to_numeric(B[c], errors='coerce')
B = B.dropna(subset=['athlete_id', 'team_id', 'game_id'])
ps = B.category.eq('passing')
B['p_att'] = B.att.where(ps, 0).fillna(0); B['p_yds'] = B.passingYards.where(ps, 0).fillna(0)
B['p_td'] = B.passingTouchdowns.where(ps, 0).fillna(0); B['p_int'] = B.interceptions.where(ps, 0).fillna(0)
B['skill'] = (B.rushingYards.where(B.category.eq('rushing'), 0).fillna(0) + B.receivingYards.where(B.category.eq('receiving'), 0).fillna(0)).clip(lower=0)
tot = B.groupby(['season', 'athlete_id', 'team_id'])[['p_att', 'p_yds', 'p_td', 'p_int', 'skill']].sum().reset_index()

# team strength at the end of each season: last postgame Elo (CFBD, where present)
tg = pd.concat([g[['season', 'date', 'home_id', 'home_postgame_elo']].set_axis(['season', 'date', 'team_id', 'elo'], axis=1),
                g[['season', 'date', 'away_id', 'away_postgame_elo']].set_axis(['season', 'date', 'team_id', 'elo'], axis=1)]).dropna(subset=['elo'])
end_elo = tg.sort_values('date').groupby(['season', 'team_id']).elo.last()
lvl = lambda s, t: (end_elo.get((s, t), 1300.0) - 1500.0) / 100.0      # teams with no Elo (FCS, lower) count as ~1300

# this season's roster: players in each team's first 3 games
sch = pd.concat([g[['game_id', 'season', 'date', 'home_id']].rename(columns={'home_id': 'team_id'}),
                 g[['game_id', 'season', 'date', 'away_id']].rename(columns={'away_id': 'team_id'})]).sort_values('date')
sch['n'] = sch.groupby(['season', 'team_id']).cumcount()
first3 = B.merge(sch[sch.n < 3][['game_id', 'team_id']], on=['game_id', 'team_id'])
qb_now = first3.groupby(['season', 'team_id', 'athlete_id']).p_att.sum().reset_index()
roster = first3[['season', 'team_id', 'athlete_id']].drop_duplicates()

rows = []
for s in sorted(roster.season.unique()):
    last = tot[tot.season == s - 1]
    if last.empty: continue
    by_ath = last.sort_values('p_att').groupby('athlete_id').agg(p_att=('p_att', 'sum'), p_yds=('p_yds', 'sum'), p_td=('p_td', 'sum'), p_int=('p_int', 'sum'),
                                                               skill=('skill', 'sum'), team=('team_id', 'last'))
    anya = (by_ath.p_yds + 20 * by_ath.p_td - 45 * by_ath.p_int)
    lg_anya = anya[by_ath.p_att >= 100].sum() / by_ath.p_att[by_ath.p_att >= 100].sum()
    lg_skill = last.groupby('team_id').skill.sum().mean()
    for t, grp in roster[roster.season == s].groupby('team_id'):
        q = qb_now[(qb_now.season == s) & (qb_now.team_id == t)].sort_values('p_att')
        qb = q.athlete_id.iloc[-1] if len(q) and q.p_att.iloc[-1] > 0 else None
        r = dict(season=s, team_id=t)
        if qb is not None and qb in by_ath.index and by_ath.p_att[qb] > 0:
            a = by_ath.loc[qb]
            r.update(qb_anya=(anya[qb] + 150 * lg_anya) / (a.p_att + 150) - lg_anya, qb_lvl=lvl(s - 1, a.team), qb_new=int(a.team != t), qb_none=0)
        else:
            r.update(qb_anya=0.0, qb_lvl=0.0, qb_new=0, qb_none=1)
        m = by_ath.reindex(grp.athlete_id.values).dropna(subset=['skill'])
        sc = np.array([1 + 0.1 * lvl(s - 1, tm) for tm in m.team])               # Elo-scaled yards
        r['avail_skill'] = float((m.skill.to_numpy() * sc).sum() / lg_skill)
        r['xfer_skill'] = float((m.skill.to_numpy() * sc * (m.team.to_numpy() != t)).sum() / lg_skill)
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(out, index=False)
print(R.groupby('season')[['qb_anya', 'qb_new', 'qb_none', 'avail_skill', 'xfer_skill']].mean().round(3).to_string()); print(len(R), 'team-seasons')
