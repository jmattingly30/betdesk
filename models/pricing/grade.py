"""Grade the price check (J 2026-11-25, desk simplification step 2). Reads history/priced.json.gz (every priced row, Pass
included: first snapshot = when the bet first showed, last = the last pregame snapshot, our stand-in for the close) and
grades each finished one at its FIRST line and price, the way J would have bet it.

  result   W / L / P at the first line; units at the first price (1 unit risked)
  clv      first price's payout x the books' no-vig chance at the last pregame snapshot - 1 (same line only; a line that moved
           shows clv_pts instead: points gained, games only)
  buckets  staked     sp|market, rows first labelled Bet (feeds auto-demote: 50+ bets, ROI -5% or worse, CLV under +1%)
           label      sp|market|label
           conf       All|N and sp|N, the 1-5 confidence rating on Bets and Leans
           guard      guard|would_be (far, long, off_line, one_sided, far_proj) - J: grade what the guards hold back
           big_dog    sp|market|label for big underdogs
           timing     sp|market|open/close
  stated vs realized: mean EV we showed vs ROI, per bucket

python3 models/pricing/grade.py -> models/pricing/history/grade.json + app/data/price_grades.json (the monitor reads it, step 3)
Finals: nflverse / sportsdataverse schedules (as models/grading/grade.py), NHL schedule, MLB finals history.
Props: player box scores through models/grading/grade.py results()."""
import json, os, sys, gzip, io, math, urllib.request, datetime as dt
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(f"{HERE}/../.."); APP = f"{ROOT}/app/data"
sys.path.insert(0, f"{ROOT}/models/grading"); sys.path.insert(0, f"{ROOT}/models/model-b")
import grade as G          # models/grading/grade.py: finals(), fetch(), results() (its main is guarded)
import devig as DV
HF, OUT = f"{HERE}/history/priced.json.gz", f"{HERE}/history/grade.json"
NHL_SCHED = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nhl_schedules/nhl_schedule_2027.parquet'
NOW = dt.datetime.now(dt.timezone.utc)

def nhl_finals():
    try:
        with urllib.request.urlopen(NHL_SCHED, timeout=90) as r: s = pd.read_parquet(io.BytesIO(r.read()))
    except Exception as e:
        print('NHL finals not available:', e); return {}
    s = s[s.game_state.isin(['OFF', 'FINAL'])]
    return {str(int(g)): (float(h), float(a)) for g, h, a in zip(s.game_id, s.home_score, s.away_score)}

def game_result(r, y):
    """W/L/P at the FIRST line for a game row; y = (home, away)"""
    yh, ya = y; f = r['first']; line = f['line']
    if r['market'] == 'spread': v = (yh - ya if r['side'] == 'home' else ya - yh) + line
    elif r['market'] == 'total': v = (yh + ya - line) if r['side'] == 'Over' else (line - yh - ya)
    else: v = (yh - ya) if r['side'] == 'home' else (ya - yh)
    return 'W' if v > 0 else 'L' if v < 0 else 'P'

def prop_result(r, RES):
    res = RES.get(r['sp'], {}); n = r['player'] + (' (G)' if r['sp'] == 'NHL' and r['stat'] == 'saves' else '')
    gk = (r.get('d') or '')[:10] if r['sp'] == 'NBA' else str(r['game'])
    v = res.get((gk, r['team'], n), {}).get(r['stat'])
    if v is None: return None, None
    L = r['first']['line']
    return ('P' if v == L else 'W' if (v > L) == (r['side'] == 'Over') else 'L'), v

def clv(r):
    """(price CLV, points CLV): first price vs the last pregame snapshot's fair chance at the same line"""
    f = r['first']
    if r.get('made') == f.get('made'): return None, None          # only one snapshot: no later market seen
    if r['line'] == f['line'] and r.get('fair') is not None:
        return round(DV.dec_from_am(f['price']) * r['fair'] - 1, 4), None
    if r['market'] in ('spread', 'total') and r['line'] is not None and f['line'] is not None:
        return None, round(f['line'] - r['line'], 2)             # a later line we'd need to lay more / get less = we beat it
    return None, None

def agg(R):
    w, l, p = (sum(x['res'] == s for x in R) for s in 'WLP'); risk = w + l; u = sum(x['u'] for x in R)
    c = [x['clv'] for x in R if x['clv'] is not None]; cp = [x['clv_pts'] for x in R if x['clv_pts'] is not None]
    return dict(n=len(R), w=w, l=l, p=p, units=round(u, 2), roi=round(u / risk, 4) if risk else None,
                ev=round(float(np.mean([x['ev'] for x in R])), 4), clv=round(float(np.mean(c)), 4) if c else None, nclv=len(c),
                clv_pts=round(float(np.mean(cp)), 2) if cp else None)

def main():
    H = json.load(gzip.open(HF, 'rt')) if os.path.exists(HF) else {}
    if not H: print('no priced history yet'); return
    sps = {r['sp'] for r in H.values()}
    FIN = {sp: (nhl_finals() if sp == 'NHL' else G.finals(sp)) for sp in sps if sp in ('NFL', 'NBA', 'CFB', 'CBB', 'MLB', 'NHL')}
    RES = {}
    for sp in ('NFL', 'NBA', 'NHL'):
        if any(r['sp'] == sp and r['market'] == 'prop' for r in H.values()): RES[sp] = G.results(sp, G.fetch(sp))
    rows, pending = [], 0
    for key, r in H.items():
        f = r.get('first') or {}
        if f.get('price') is None or f.get('line') is None and r['market'] != 'ml': continue
        if r['market'] == 'prop': res, act = prop_result(r, RES)
        else:
            y = FIN.get(r['sp'], {}).get(str(r['game'])); res, act = (game_result(r, y), list(y)) if y else (None, None)
        if res is None:
            pending += dt.datetime.fromisoformat(r['k'].replace('Z', '+00:00')) > NOW - dt.timedelta(days=4); continue
        d = DV.dec_from_am(f['price']); u = {'W': d - 1, 'L': -1.0, 'P': 0.0}[res]; c, cp = clv(r)
        rows.append(dict(key=key, sp=r['sp'], market=r['market'], stat=r.get('stat'), side=r['side'], d=r['k'][:10], label=f.get('label'),
                         would_be=f.get('would_be') or f.get('label'), guards=f.get('guards') or [], big_dog=bool(r.get('big_dog')), timing=f.get('timing'),
                         line=f['line'], price=f['price'], ours=f.get('ours'), fair=f.get('fair'), trusted=f.get('trusted'), trust=f.get('trust'),
                         ev=f.get('ev') or 0.0, conf=f.get('conf', r.get('conf')), close_fair=r.get('fair'), close_line=r.get('line'), close_ours=r.get('ours'),
                         res=res, act=act, u=round(u, 3), clv=c, clv_pts=cp))
    B = {'staked': {}, 'label': {}, 'guard': {}, 'big_dog': {}, 'timing': {}, 'conf': {}}
    def add(kind, k, x): B[kind].setdefault(k, []).append(x)
    for x in rows:
        if x['label'] == 'Bet': add('staked', f"{x['sp']}|{x['market']}", x)
        add('label', f"{x['sp']}|{x['market']}|{x['label']}", x)
        for g in x['guards']:
            if x['would_be'] != 'Pass': add('guard', f"{g}|{x['would_be']}", x)
        if x['big_dog']: add('big_dog', f"{x['sp']}|{x['market']}|{x['would_be']}", x)
        if x['timing']: add('timing', f"{x['sp']}|{x['market']}|{x['timing']}", x)
        # confidence rating (J 2026-11-25): do 5s beat 3s? all sports together, then by sport; unstaked guarded rows left out
        if x['conf'] and not x['guards']: add('conf', f"All|{x['conf']}", x); add('conf', f"{x['sp']}|{x['conf']}", x)
    out = dict(made=NOW.strftime('%Y-%m-%d %H:%M UTC'), graded=len(rows), pending=pending,
               **{k: {kk: agg(v) for kk, v in sorted(d.items())} for k, d in B.items()})
    json.dump(out, open(OUT, 'w'), indent=1)
    # graded rows for the refit (trust needs every row with its snapshot chances) and the monitor
    json.dump(rows, gzip.open(f"{HERE}/history/graded.json.gz", 'wt'), separators=(',', ':'))
    json.dump(dict(out, rows=sorted(rows, key=lambda x: x['d'], reverse=True)[:400]), open(f"{APP}/price_grades.json", 'w'), separators=(',', ':'))
    print('price check graded', len(rows), 'pending', pending)
    for k, v in out['staked'].items(): print(' Bet', k, v)
    for k, v in out['guard'].items(): print(' guard', k, v)

if __name__ == '__main__':
    main()
