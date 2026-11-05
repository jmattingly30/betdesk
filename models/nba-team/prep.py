"""Build the NBA team-model tables.

usage: python3 prep.py <sbr_json>     (run from models/nba-team; reads ../team-strength/data and ../nba-player/data)
<sbr_json> = raw.githubusercontent.com/flancast90/sportsbookreview-scraper/main/data/nba_archive_10Y.json
             (Sportsbook Review archive: opening + closing spreads and totals, 2011-12 .. 2021-22)
Writes data/games.parquet   one row per game: scores, possessions, rest, closing line (ESPN/Odds API, 2018+),
                             SBR open/close spread + total (2012-2022)    line convention: expected HOME margin (+ = home favoured)
       data/team_games.parquet  two rows per game (offense team vs defense opp) with points and game possessions"""
import sys, json, numpy as np, pandas as pd
TS, NP = '../team-strength/data', '../nba-player/data'
g = pd.read_parquet(f'{TS}/nba_games.parquet')
tg = pd.read_parquet(f'{TS}/nba_team_games.parquet')
NBA = {'ATL','BKN','BOS','CHA','CHI','CLE','DAL','DEN','DET','GS','HOU','IND','LAC','LAL','MEM','MIA','MIL','MIN','NJ','NO','NY',
       'OKC','ORL','PHI','PHX','POR','SA','SAC','SEA','TOR','UTAH','WSH'}
g = g[g.home.isin(NBA) & g.away.isin(NBA)].copy()
# game possessions = average of the two teams' estimates
pos = tg.groupby('game_id').poss.mean()
g['poss'] = g.game_id.map(pos); g = g[g.poss.notna()]
g['total'] = g.home_score + g.away_score
g['neutral'] = g.neutral.astype(bool)

# ---- Sportsbook Review open/close lines, matched by date (+-1 day) and final score
s = pd.DataFrame(json.load(open(sys.argv[1])))
for c in ['home_final', 'away_final', 'home_open_spread', 'home_close_spread', 'open_over_under', 'close_over_under', 'home_close_ml', 'away_close_ml']:
    s[c] = pd.to_numeric(s[c], errors='coerce')
s['day'] = pd.to_datetime(s.date.astype('Int64').astype(str), format='%Y%m%d', errors='coerce')
s = s[s.day.notna() & s.home_final.notna()]
s['sbr_open'] = -s.home_open_spread; s['sbr_close'] = -s.home_close_spread
s['sbr_total_open'] = s.open_over_under; s['sbr_total_close'] = s.close_over_under
s['sbr_ml_home'], s['sbr_ml_away'] = s.home_close_ml, s.away_close_ml
keep = ['sbr_open', 'sbr_close', 'sbr_total_open', 'sbr_total_close', 'sbr_ml_home', 'sbr_ml_away']
m = []
for off in [0, -1, 1]:
    t = s.assign(day=s.day + pd.Timedelta(days=off))[['day', 'home_final', 'away_final'] + keep]
    m.append(g[['game_id', 'day', 'home_score', 'away_score']].merge(t, left_on=['day', 'home_score', 'away_score'], right_on=['day', 'home_final', 'away_final']))
m = pd.concat(m).drop_duplicates('game_id')
dup = m.groupby(['day', 'home_score', 'away_score']).game_id.transform('size') > 1
m = m[~dup]
g = g.merge(m[['game_id'] + keep], on='game_id', how='left')
# SBR data errors: totals stored in the spread column and vice versa, 'pk' etc. Keep plausible numbers only.
for c in ['sbr_open', 'sbr_close']: g.loc[g[c].abs() > 25, c] = np.nan
for c in ['sbr_total_open', 'sbr_total_close']: g.loc[~g[c].between(170, 260), c] = np.nan
bad = (g.sbr_open - g.sbr_close).abs() > 8; g.loc[bad, ['sbr_open']] = np.nan
badt = (g.sbr_total_open - g.sbr_total_close).abs() > 15; g.loc[badt, 'sbr_total_open'] = np.nan
print('SBR matched', len(m), 'of', len(s), '| usable openers', int(g.sbr_open.notna().sum()), '| removed', int(bad.sum()), 'spread', int(badt.sum()), 'total')
both = g.line.notna() & g.sbr_close.notna()
print('SBR close vs ESPN/OddsAPI close: agree within 0.5 on %.1f%% of %d games' % (100 * ((g.line - g.sbr_close).abs() <= 0.5)[both].mean(), both.sum()))
g['close'] = g.line.fillna(g.sbr_close); g['total_close'] = g.total_line.fillna(g.sbr_total_close)
g = g.sort_values(['date', 'game_id']).reset_index(drop=True)
cols = ['game_id', 'season', 'date', 'day', 'home', 'away', 'neutral', 'playoff', 'home_score', 'away_score', 'margin', 'total', 'poss',
        'home_rest', 'away_rest', 'line', 'total_line', 'close', 'total_close'] + keep
g[cols].to_parquet('data/games.parquet')
print(g[g.close.notna()].groupby('season').agg(n=('game_id', 'size'), open=('sbr_open', 'count')).T.to_string())

t = tg[tg.game_id.isin(g.game_id)][['game_id', 'team', 'opp', 'pts']].copy()
t = t.merge(g[['game_id', 'season', 'date', 'home', 'neutral', 'poss']], on='game_id')
t['ha'] = np.where(t.team == t.home, 'home', 'away')
t.drop(columns='home').to_parquet('data/team_games.parquet')
print('team-games', len(t))
