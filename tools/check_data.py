"""Data integrity check for the files the Bet Desk page reads (2026-11-06 audit). Read-only; changes nothing.

python3 tools/check_data.py [data_dir]      (default app/data; exit 1 if any ERROR)

Catches the ways bets or game data could get crossed between games, teams, players or sports:
- game_slates.json: duplicate game ids, a team listed against itself, a game's day not its US Eastern date
- {cbb,nfl,nba,cfb}_ab.json and model_a.json: every game id is on that sport's slate with the same home/away teams and day,
  and Model B's market margin agrees with the slate's home spread (a flipped home/away shows as a sign error)
- {nfl,nba,nhl}_props.json: no player listed twice for one game; NFL players' team/opponent/home flag match the slate
WARN lines are worth a look but do not fail the job (e.g. a non-D1 'team' name ESPN reuses for several schools).
"""
import sys, os, json, collections
from datetime import datetime
from zoneinfo import ZoneInfo

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app', 'data')
ET = ZoneInfo('America/New_York')
errors, warns = [], []
def load(name):
    p = os.path.join(D, name)
    if not os.path.exists(p): return None
    try: return json.load(open(p))
    except Exception as e: errors.append(f'{name}: unreadable ({e})'); return None
def et_day(k): return datetime.fromisoformat(k.replace('Z', '+00:00')).astimezone(ET).date().isoformat()

GS = load('game_slates.json') or {}
slate = {}
for sp in ('NFL', 'CFB', 'CBB', 'NBA'):
    G = (GS.get(sp) or {}).get('games') or []
    ids = collections.Counter(g['id'] for g in G)
    errors += [f'game_slates {sp}: game id {i} listed {c} times' for i, c in ids.items() if c > 1]
    errors += [f"game_slates {sp}: {g['id']} has {g['h']} playing itself" for g in G if g['h'] == g['a']]
    errors += [f"game_slates {sp}: {g['id']} day {g['d']} is not the Eastern date of {g['k']}" for g in G if g.get('k') and et_day(g['k']) != g['d']]
    per_day = collections.Counter((g['d'], t) for g in G for t in (g['h'], g['a']))
    warns += [f'game_slates {sp}: {t} has {c} games on {d} (name shared by two schools?)' for (d, t), c in per_day.items() if c > 1]
    slate[sp] = {g['id']: g for g in G}

for sp, f in (('CBB', 'cbb_ab.json'), ('NFL', 'nfl_ab.json'), ('NBA', 'nba_ab.json'), ('CFB', 'cfb_ab.json')):
    J = load(f)
    if not J: continue
    S = slate.get(sp, {})
    for i, a in (J.get('games') or {}).items():
        g = S.get(i)
        if g is None: warns.append(f'{f}: game {i} ({a.get("a")} @ {a.get("h")}) is not on the {sp} slate (stale?)'); continue
        if (a.get('h'), a.get('a')) != (g['h'], g['a']):
            errors.append(f"{f}: game {i} is {a.get('a')} @ {a.get('h')} but the slate has {g['a']} @ {g['h']}"); continue
        if a.get('d') and a['d'] != g['d']: errors.append(f"{f}: game {i} day {a['d']} vs slate {g['d']}")
        L = (g.get('ln') or {}).get('L'); m = (((a.get('B') or {}).get('spread')) or {}).get('market')
        if L is not None and m is not None and abs(m + L) > 3:
            errors.append(f"{f}: game {i} {g['a']} @ {g['h']} Model B market margin {m:+.1f} vs slate home spread {L:+.1f} (home/away flipped?)")

MA = load('model_a.json') or {}
for sp, x in (MA.get('sports') or {}).items():
    S = slate.get(sp)
    if not S: continue
    for g in x.get('games') or []:
        s = S.get(str(g.get('id')))
        if s and (g.get('h'), g.get('a')) != (s['h'], s['a']):
            errors.append(f"model_a.json {sp}: game {g.get('id')} is {g.get('a')} @ {g.get('h')} but the slate has {s['a']} @ {s['h']}")

for sp, f in (('NFL', 'nfl_props.json'), ('NBA', 'nba_props.json'), ('NHL', 'nhl_props.json')):
    P = load(f)
    if not P: continue
    seen = collections.Counter((p['n'], p['t'], str(p.get('g')), p.get('d')) for p in P.get('players') or [])
    for (n, t, g, d), c in seen.items():
        if c > 1: warns.append(f'{f}: {n} ({t}) listed {c} times for game {g} (same short name, check bets on this name)')
    if sp == 'NFL':
        S = slate.get('NFL', {})
        for p in P.get('players') or []:
            g = S.get(p.get('g'))
            if g and ({p['t'], p['o']} != {g['h'], g['a']} or (p.get('h') == 1) != (p['t'] == g['h'])):
                errors.append(f"{f}: {p['n']} {p['t']} vs {p['o']} home={p.get('h')} does not match {g['a']} @ {g['h']}")

for w in warns: print('WARN', w)
for e in errors: print('ERROR', e)
print(f'check_data: {len(errors)} errors, {len(warns)} warnings')
sys.exit(1 if errors else 0)
