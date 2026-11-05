"""Parse ESPN pickcenter lines (fetched from hoopR-mbb-raw game JSONs) into one row per game.
line   = expected home margin implied by the closing spread (home -7 -> +7). Source priority: consensus, teamrankings, sportsbook.
line_cons / line_tr = consensus and TeamRankings spreads separately (they disagree on ~1/3 of games, see game_model.py).
line_open / total_open = DraftKings opening numbers (only 2025-26 carries them).
Also keeps numberFire's projected scores and ESPN BPI win probability as public-model benchmarks.
usage: python3 prep_lines.py <lines.jsonl> <out parquet>"""
import sys, json, numpy as np, pandas as pd
src, out = sys.argv[1:3]
def num(x):
    try: return float(str(x).replace('o', '').replace('u', '').replace('+', '')) if x not in (None, '', 'EVEN') else (100.0 if x == 'EVEN' else np.nan)
    except ValueError: return np.nan
rows = {}
for l in open(src):
    r = json.loads(l); g = r['g']
    if 'err' in r: continue
    pc = {p['provider']['name']: p for p in r.get('pc', [])}
    row = {'game_id': g}
    for name in ['consensus', 'teamrankings', 'Caesars Sportsbook (New Jersey)', 'Caesars Sportsbook (Pennsylvania)', 'DraftKings']:
        p = pc.get(name)
        if p and p.get('spread') is not None and 'line' not in row:
            row['line'] = -float(p['spread']); row['line_src'] = name
        if p and p.get('overUnder') is not None and 'total_line' not in row:
            row['total_line'] = float(p['overUnder'])
        if p and 'ml_home' not in row and p.get('homeTeamOdds', {}).get('moneyLine') is not None and p.get('awayTeamOdds', {}).get('moneyLine') is not None:
            row['ml_home'] = float(p['homeTeamOdds']['moneyLine']); row['ml_away'] = float(p['awayTeamOdds']['moneyLine'])
    for name, col in [('consensus', 'line_cons'), ('teamrankings', 'line_tr')]:
        p = pc.get(name)
        if p and p.get('spread') is not None: row[col] = -float(p['spread'])
    dk = pc.get('DraftKings')
    if dk and dk.get('pointSpread'):
        ps = dk['pointSpread'].get('home', {})
        row['line_open'] = -num(ps.get('open', {}).get('line')); row['line_close_dk'] = -num(ps.get('close', {}).get('line'))
        t = dk.get('total', {}).get('over', {})
        row['total_open'] = num(t.get('open', {}).get('line')); row['total_close_dk'] = num(t.get('close', {}).get('line'))
        ml = dk.get('moneyline', {})
        row['ml_home_open'] = num(ml.get('home', {}).get('open', {}).get('odds')); row['ml_away_open'] = num(ml.get('away', {}).get('open', {}).get('odds'))
    nf = pc.get('numberfire')
    if nf:
        row['nf_home'] = nf['homeTeamOdds'].get('averageScore'); row['nf_away'] = nf['awayTeamOdds'].get('averageScore')
    pr = r.get('pred') or {}
    try: row['bpi_home'] = float(pr['homeTeam']['gameProjection']) / 100
    except Exception: pass
    rows[g] = row
d = pd.DataFrame(rows.values())
d.to_parquet(out)
print(d.shape); print(d.notna().sum().to_string()); print(d.line_src.value_counts().to_string())
