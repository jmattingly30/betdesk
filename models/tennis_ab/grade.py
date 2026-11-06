"""Grade tennis Model A + B from history/runs.jsonl against tennisexplorer results (match id = tex_mid; one results page per
day, 3 s apart, pages cached gzipped in history/pages/). Model A: total games error, each player's games error, winner pick,
set score. Model B: paper result of tier B leans at the logged price (first time each match was seen). Writes history/grade.json."""
import os, sys, json, gzip, time, datetime as dt, urllib.request
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'tennis-golf-research'))
import tex_scores
PG = f'{HERE}/history/pages'; os.makedirs(PG, exist_ok=True)
def page(d):
    p = f'{PG}/{d:%Y%m%d}.html.gz'
    if os.path.exists(p) and d < dt.date.today() - dt.timedelta(days=1): return gzip.open(p, 'rt').read()
    url = f'https://www.tennisexplorer.com/results/?type=all&year={d.year}&month={d.month:02d}&day={d.day:02d}'
    s = urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60).read().decode('utf-8', 'ignore')
    gzip.open(p, 'wt').write(s); time.sleep(3); return s
runs = [json.loads(l) for l in open(f'{HERE}/history/runs.jsonl')] if os.path.exists(f'{HERE}/history/runs.jsonl') else []
first = {}
for r in runs:
    if r.get('tex_mid') and r['tex_mid'] not in first: first[r['tex_mid']] = r
days = sorted({dt.date.fromisoformat((r.get('start') or r['made'])[:10]) for r in first.values()})
today = dt.datetime.utcnow().date()
res = {}
for d in days:
    for dd in (d - dt.timedelta(days=1), d, d + dt.timedelta(days=1)):
        if dd >= today: continue
        try:
            for x in tex_scores.parse(page(dd), dd): res[str(x['mid'])] = x
        except Exception as e: print('page', dd, e)
rows = []
for mid, r in first.items():
    x = res.get(mid)
    if not x or not x['nsets'] or str(x['sets_a']) == str(x['sets_b']): continue
    # tex lists the winner first; our p1 is the tex name or the OddsPapi name (match on surname tokens)
    sur = lambda s: set(t for t in s.lower().replace(',', ' ').replace('.', ' ').split() if len(t) > 1)
    p1_is_a = bool(sur(r['p1']) & sur(x['pa'])) or not (sur(r['p1']) & sur(x['pb']))
    g1, g2 = (x['games_a'], x['games_b']) if p1_is_a else (x['games_b'], x['games_a'])
    p1_won = int(x['sets_a']) > int(x['sets_b']) if p1_is_a else int(x['sets_b']) > int(x['sets_a'])
    rows.append(dict(mid=mid, level=f"{r['sex']}|{r['level']}", tot=g1 + g2, g1=g1, g2=g2, p1_won=p1_won, A=r['A'], B=r['B']))
out = dict(made=dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), n=len(rows))
if rows:
    A = lambda f: [f(z) for z in rows]
    out['model_a'] = dict(tot_mae=round(float(np.mean(A(lambda z: abs(z['A']['tot_med'] - z['tot'])))), 2),
                          tot_mae_mean=round(float(np.mean(A(lambda z: abs(z['A']['tot_mean'] - z['tot'])))), 2),
                          games_mae=round(float(np.mean(A(lambda z: (abs(z['A']['games_p1'] - z['g1']) + abs(z['A']['games_p2'] - z['g2'])) / 2))), 2),
                          winner_right=round(float(np.mean(A(lambda z: (z['A']['win_p1'] > .5) == z['p1_won']))), 3),
                          by_level={lv: len([z for z in rows if z['level'] == lv]) for lv in sorted({z['level'] for z in rows})})
    bets = [z for z in rows if z['B'].get('tier') == 'B' and z['B'].get('price')]
    pnl = [(z['B']['price'] - 1) if ((z['B']['side'] == 1) == z['p1_won']) else -1 for z in bets]
    out['model_b_leans'] = dict(n=len(bets), units=round(float(sum(pnl)), 2), roi=round(float(np.mean(pnl)) * 100, 1) if pnl else None)
    # tier C passes by their unproven rec (J 2026-11-06), paper at the logged price
    sys.path.insert(0, os.path.join(HERE, '..', 'model-b')); import unproven
    for rec in ('play', 'watch', 'skip'):
        U = [z for z in rows if (z['B'].get('u') or unproven.rate_tennis(dict(A=z['A'], B=z['B'], sex=z['level'][0])) or {}).get('rec') == rec]
        pu = [(z['B']['price'] - 1) if ((z['B']['side'] == 1) == z['p1_won']) else -1 for z in U]
        out[f'unproven_{rec}'] = dict(n=len(U), units=round(float(sum(pu)), 2), roi=round(float(np.mean(pu)) * 100, 1) if pu else None)
json.dump(out, open(f'{HERE}/history/grade.json', 'w'), indent=1)
print(json.dumps(out))
