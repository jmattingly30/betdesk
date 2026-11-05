"""Flatten the SportsBookReview MLB odds dataset (ArnavSaraogi/mlb-odds-scraper, 2021-03-20..2025-08-16)
into one row per game with consensus no-vig opening and closing home win probabilities and totals."""
import json, sys, numpy as np, pandas as pd
src, out = sys.argv[1], sys.argv[2]
def p(ml):  # american -> implied prob
    return 100/(ml+100) if ml > 0 else -ml/(-ml+100)
rows = []
for date, games in json.load(open(src)).items():
    for g in games:
        gv = g['gameView']; o = g.get('odds') or {}
        if gv.get('gameType') != 'R' or gv.get('homeTeamScore') is None: continue
        r = dict(date=date, start=gv['startDate'], home=gv['homeTeam']['shortName'], away=gv['awayTeam']['shortName'],
                 hs=gv['homeTeamScore'], as_=gv['awayTeamScore'], status=gv.get('gameStatusText'))
        for when, key in (('open', 'openingLine'), ('close', 'currentLine')):
            probs = []
            for b in o.get('moneyline') or []:
                l = b.get(key) or {}
                h, a = l.get('homeOdds'), l.get('awayOdds')
                if h and a:
                    ph, pa = p(h), p(a); probs.append(ph/(ph+pa))
            r[f'ph_{when}'] = np.median(probs) if probs else np.nan
            r[f'nbooks_{when}'] = len(probs)
            tots = [(b.get(key) or {}).get('total') for b in o.get('totals') or []]
            tots = [t for t in tots if t]
            r[f'total_{when}'] = np.median(tots) if tots else np.nan
            dk = [b for b in o.get('moneyline') or [] if b['sportsbook'] == 'draftkings']
            if dk and (dk[0].get(key) or {}).get('homeOdds'):
                r[f'dk_home_{when}'] = dk[0][key]['homeOdds']; r[f'dk_away_{when}'] = dk[0][key]['awayOdds']
        rows.append(r)
d = pd.DataFrame(rows)
d.to_csv(out, index=False)
print(d.shape, d[['ph_open','ph_close','total_open','total_close']].notna().mean().round(3).to_dict())
