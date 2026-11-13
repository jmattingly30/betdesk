"""Drop model rows whose game id now belongs to a different matchup on the slate (2026-11-13).

python3 tools/drop_crossed_games.py [data_dir]      (default app/data; runs in tools/ci.sh publish(), before the data check)

Why: the models project games from the morning's schedule, but ESPN sometimes re-uses or fills in a game id later in the day
(a tournament bracket set after round one, a placeholder "TBD" filled in, home/away corrected). The noon/night pulls rebuild
the slate from ESPN but keep the morning's Model A numbers, so a row like "Towson @ Morgan State" sat under the id that is now
"Morgan State @ Loyola Maryland" and tools/check_data.py failed every run until the next morning (11-12 and 11-13 runs).
That projection is for the wrong game, so the honest fix is to take it off the desk until the next model run re-projects it.

What counts as crossed: the id is on that sport's slate, the slate's game has NOT started, and the row's home/away teams differ
from the slate's (any difference, including a pre-start home/away swap). A started game whose sides ESPN swapped afterwards
(neutral site) is left alone, as tools/check_data.py already warns about it. Rows whose id is not on the slate are left alone.
Each drop is printed as a WARN line so it shows in the job log.
"""
import sys, os, json
from datetime import datetime, timezone

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'app', 'data')


def load(name):
    try: return json.load(open(os.path.join(D, name)))
    except Exception: return None


def started(g):
    try: return datetime.fromisoformat(g['k'].replace('Z', '+00:00')) <= datetime.now(timezone.utc)
    except Exception: return False


GS = load('game_slates.json') or {}
slate = {sp: {str(g['id']): g for g in (GS.get(sp) or {}).get('games') or []} for sp in ('NFL', 'CFB', 'CBB', 'NBA')}


def crossed(sp, gid, row):
    s = slate.get(sp, {}).get(str(gid))
    if not s or not isinstance(row, dict) or started(s): return None
    if (row.get('h'), row.get('a')) == (s['h'], s['a']): return None
    return f"{row.get('a')} @ {row.get('h')} but the slate now has {s['a']} @ {s['h']}"


total = 0
def save(name, data, n):
    global total
    if n:
        json.dump(data, open(os.path.join(D, name), 'w'), separators=(',', ':'), ensure_ascii=False)
        total += n


# files keyed {id: row}
for sp, name in (('CBB', 'cbb_ab.json'), ('NFL', 'nfl_ab.json'), ('NBA', 'nba_ab.json'), ('CFB', 'cfb_ab.json'),
                 ('CBB', 'sims_cbb.json'), ('NFL', 'sims_nfl.json'), ('NBA', 'sims_nba.json'), ('CFB', 'sims_cfb.json')):
    J = load(name)
    G = J and J.get('games')
    if not isinstance(G, dict): continue
    n = 0
    for gid in list(G):
        why = crossed(sp, gid, G[gid])
        if why:
            print(f'WARN dropped {name} game {gid}: model row is {why} (next model run re-projects it)')
            del G[gid]; n += 1
    save(name, J, n)

# model_a.json: {sports: {SP: {games: [row with id]}}}
MA = load('model_a.json')
if MA:
    n = 0
    for sp, x in (MA.get('sports') or {}).items():
        rows = x.get('games') if isinstance(x, dict) else None
        if not isinstance(rows, list): continue
        keep = []
        for r in rows:
            why = crossed(sp, r.get('id') if isinstance(r, dict) else None, r)
            if why: print(f"WARN dropped model_a.json {sp} game {r.get('id')}: model row is {why} (next model run re-projects it)"); n += 1
            else: keep.append(r)
        x['games'] = keep
    save('model_a.json', MA, n)

print(f'drop_crossed_games: {total} row(s) dropped')
