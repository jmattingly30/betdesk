"""Head-coach grades for the Bet Desk NFL card (coach_grades.py, J 2026-10-11). Grade = career mean (actual margin minus
closing spread), every game he has coached since 2008, shrunk with 48 games of 0. Backtest 2010-2025: margin vs close
coef +0.50 per point of grade gap (t 3.0), weaker 2020-25 (+0.29, t 1.2), so the app uses 0.5 x the gap (raised after the big-gap test).
usage: python3 nfl/export_coach_grades.py [out.json]      (current coach per team from the 2026 schedule)"""
import sys, os, json, numpy as np, pandas as pd
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = sys.argv[1] if len(sys.argv) > 1 else os.path.normpath(f'{H}/../../app/data/nfl_coaches.json')
g = pd.read_parquet(f'{H}/data/nfl_games_features.parquet')
g = g[g.margin.notna() & g.spread_line.notna() & (g.season >= 2008)]
r = pd.concat([pd.DataFrame({'coach': g.home_coach, 'r': g.margin - g.spread_line}), pd.DataFrame({'coach': g.away_coach, 'r': -(g.margin - g.spread_line)})])
c = r.groupby('coach').r.agg(['sum', 'size']); c['grade'] = c['sum'] / (c['size'] + 48)
cur = g[g.season == g.season.max()].sort_values('gameday')
team = {}
for side in ('home', 'away'):
    for t, co in zip(cur[f'{side}_team'], cur[f'{side}_coach']): team[t] = co
doc = dict(made=str(g.gameday.max()), weight=0.5, note='career margin minus closing spread per game, shrunk (48 games of 0)',
           teams={t: [co, round(float(c.grade.get(co, 0.0)), 2), int(c['size'].get(co, 0))] for t, co in sorted(team.items())})
json.dump(doc, open(out, 'w'), separators=(',', ':'))
print(len(doc['teams']), 'teams ->', out); print(sorted(doc['teams'].items(), key=lambda x: -x[1][1])[:5])
