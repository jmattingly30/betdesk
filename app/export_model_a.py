"""Model A / Model B for the Bet Desk (J 2026-10-28 Task 2, redesign 2026-10-30).
Each game card shows Model A (the score rebuild) beside Model B (the desk's current model, priced live on the card).
This writes app/data/model_a.json with the model the card does NOT already compute, per sport:
  NBA, NFL, NHL  role A = the rebuild (models/rebuild/shadow/history.json, written by shadow.py)
  CFB            role A = the rebuild v4 (app/data/cfb_live.json, from the rebuild thread's cfb_live.py) when present
  CBB            role B = the previous models/cbb model (cbb_v2.json today_hs/as); the desk card already runs CBB v2 (A)
plus the rebuild-vs-desk grading from shadow/report.json, so the Data tab can chart both models.
Run after shadow.py (morning chain): python3 app/export_model_a.py"""
import json, os, datetime as dt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
SH = f'{ROOT}/models/rebuild/shadow'; DATA = f'{HERE}/data'
now = dt.datetime.now(dt.timezone.utc)
live = lambda k: not k or dt.datetime.fromisoformat(k.replace('Z', '+00:00')) > now - dt.timedelta(hours=4)
out = dict(made=now.strftime('%Y-%m-%d %H:%M UTC'), sports={}, grade=None,
           note='A = rebuild (NBA, NFL, NHL, CFB); for CBB the desk card runs CBB v2 (A) and this file carries the previous CBB model (B).')
hist = json.load(open(f'{SH}/history.json')) if os.path.exists(f'{SH}/history.json') else {}
SRC = {'stack': 'full stack: ratings, tonight\'s projected lineup and schedule'}
for sp, label, why in (('NBA', 'Rebuild', 'possession engine: Kalman team ratings, projected lineup, rest'),
                       ('NFL', 'Rebuild', 'drive engine: Kalman unit ratings, depth-chart QB, missing regulars, dome'),
                       ('NHL', 'Rebuild', 'goal-state engine: 5v5 xG ratings, power play, goalie (margin only), rest')):
    G = [dict(id=v['id'], h=v['h'], a=v['a'], d=v.get('d'), k=v.get('k'), hs=v['hs'], **{'as': v['as']}, src=v.get('src', 'ratings'), made=v.get('made'))
         for v in hist.values() if v.get('sp') == sp and live(v.get('k'))]
    if G: out['sports'][sp] = dict(role='A', label=label, how=why, games=sorted(G, key=lambda g: (g['k'] or '', g['h'])))
cl = f'{DATA}/cfb_live.json'
if os.path.exists(cl):
    j = json.load(open(cl))
    G = [dict(id=str(g['id']), h=g['h'], a=g['a'], k=g.get('k'), hs=g['hs'], **{'as': g['as']}, v3=[g.get('v3_hs'), g.get('v3_as')], src='v4') for g in j.get('games', []) if live(g.get('k'))]
    if G: out['sports']['CFB'] = dict(role='A', label='Rebuild (CFB v4)', how='v3 plus program SRS, coach career win%, crowd size; in-season tempo for totals', made=j.get('made'), games=G)
cv = f'{DATA}/cbb_v2.json'
if os.path.exists(cv):
    j = json.load(open(cv))
    G = [dict(id=gid, h=x['h'], a=x['a'], d=x.get('d'), hs=x['today_hs'], **{'as': x['today_as']}, rb=[x.get('rb_hs'), x.get('rb_as')]) for gid, x in j['games'].items()]
    out['sports']['CBB'] = dict(role='B', label='Previous CBB model', how='models/cbb ratings (the old desk model)', made=j.get('made'), coef=j.get('coef'), games=G)
if os.path.exists(f'{SH}/report.json'): out['grade'] = json.load(open(f'{SH}/report.json'))
clean = lambda v: {k: clean(x) for k, x in v.items()} if isinstance(v, dict) else [clean(x) for x in v] if isinstance(v, list) else None if isinstance(v, float) and v != v else v   # NaN is not valid JSON in the browser
json.dump(clean(out), open(f'{DATA}/model_a.json', 'w'), separators=(',', ':'), allow_nan=False)
print('model_a.json', {sp: len(v['games']) for sp, v in out['sports'].items()}, 'grade', bool(out['grade']))
