"""One row per D1 men's college basketball game (ESPN via hoopR-mbb-data, 2008-2026 seasons) and one row per team-game
with possessions, efficiency and four factors. Season = ESPN end year (2026 = 2025-26).

usage: python3 prep_cbb.py <hoopR download dir> <out dir>"""
import os, sys, numpy as np, pandas as pd
# CBB_LAST=2027 adds the in-progress season once hoopR posts its box files (missing years are skipped)
LAST = int(os.environ.get('CBB_LAST', 2026))
YEARS = lambda a: [y for y in range(a, LAST + 1) if y < 2027 or os.path.exists(f'{sys.argv[1]}/team_box_{y}.parquet')]
d, out = sys.argv[1:3]
G, TG = [], []
for y in YEARS(2008):
    s = pd.read_parquet(f'{d}/mbb_schedule_{y}.parquet')
    s = s[s.status_type_completed.astype(bool) & s.season_type.isin([2, 3])]
    b = pd.read_parquet(f'{d}/team_box_{y}.parquet')
    b = b[b.game_id.isin(s.game_id)]
    num = ['team_score', 'opponent_team_score', 'field_goals_made', 'field_goals_attempted', 'three_point_field_goals_made',
           'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted', 'offensive_rebounds',
           'defensive_rebounds', 'total_turnovers', 'turnovers', 'assists', 'steals', 'blocks', 'fouls']
    for c in num: b[c] = pd.to_numeric(b[c], errors='coerce')
    b['tov'] = b.total_turnovers.fillna(b.turnovers)
    o = b[['game_id', 'team_id', 'opponent_team_id', 'team_home_away', 'team_score', 'field_goals_made', 'field_goals_attempted',
           'three_point_field_goals_made', 'three_point_field_goals_attempted', 'free_throws_made', 'free_throws_attempted',
           'offensive_rebounds', 'defensive_rebounds', 'tov', 'assists', 'steals', 'blocks']].copy()
    o.columns = ['game_id', 'team', 'opp', 'ha', 'pts', 'fgm', 'fga', 'tpm', 'tpa', 'ftm', 'fta', 'orb', 'drb', 'tov', 'ast', 'stl', 'blk']
    o = o.drop_duplicates(['game_id', 'team'])
    o = o[o.groupby('game_id').team.transform('size') == 2]
    stat = ['pts', 'fgm', 'fga', 'tpm', 'tpa', 'ftm', 'fta', 'orb', 'drb', 'tov']
    o = o.merge(o[['game_id', 'team'] + stat].rename(columns={'team': 'opp', **{c: 'o_' + c for c in stat}}), on=['game_id', 'opp'])
    # possessions (college convention 0.475*FTA), averaged over both teams
    o['poss_raw'] = o.fga - o.orb + o.tov + 0.475 * o.fta
    o['poss'] = (o.poss_raw + (o.o_fga - o.o_orb + o.o_tov + 0.475 * o.o_fta)) / 2
    o['season'] = y
    TG.append(o)
    g = s[['game_id', 'season', 'season_type', 'game_date', 'game_date_time', 'neutral_site', 'conference_competition',
           'home_id', 'away_id', 'home_location', 'away_location', 'home_score', 'away_score', 'home_conference_id',
           'away_conference_id', 'notes_headline', 'tournament_id']].copy()
    g = g[g.game_id.isin(o.game_id)]
    G.append(g)
g = pd.concat(G, ignore_index=True); o = pd.concat(TG, ignore_index=True)
g['date'] = pd.to_datetime(g.game_date); g['home_score'] = g.home_score.astype(float); g['away_score'] = g.away_score.astype(float)
g['margin'] = g.home_score - g.away_score; g['total'] = g.home_score + g.away_score
g['neutral'] = g.neutral_site.astype(bool)
g = g.rename(columns={'home_id': 'home', 'away_id': 'away'})
# Division I proxy: team plays >= 15 games in the season against teams that themselves play >= 15 games
cnt = pd.concat([g[['season', 'home']].rename(columns={'home': 't'}), g[['season', 'away']].rename(columns={'away': 't'})]).value_counts()
d1 = set(cnt[cnt >= 15].index)
# in-progress season (CBB_LAST beyond the last finished one): last season's D1 teams count as D1 until 15 games are in
if LAST > 2026: d1 |= {(LAST, t) for (s_, t) in list(d1) if s_ == LAST - 1}
g['d1'] = [((s_, h) in d1) and ((s_, a) in d1) for s_, h, a in zip(g.season, g.home, g.away)]
poss = o.groupby('game_id').poss.first()
g['poss'] = g.game_id.map(poss)
g = g.sort_values(['date', 'game_id']).reset_index(drop=True)
o = o.merge(g[['game_id', 'date', 'neutral', 'season_type']], on='game_id')
g.to_parquet(f'{out}/cbb_games.parquet'); o.to_parquet(f'{out}/cbb_team_games.parquet')
print('games', len(g), 'D1 v D1', g.d1.sum()); print(g[g.d1].groupby('season').agg(n=('game_id', 'size'), poss=('poss', 'mean'), pts=('total', 'mean'),
      hm=('margin', lambda x: x[~g.loc[x.index, 'neutral']].mean()), neutral=('neutral', 'mean')).round(2).to_string())
