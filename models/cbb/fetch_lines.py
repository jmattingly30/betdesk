"""Pull betting lines (ESPN pickcenter) and ESPN BPI predictor for every completed D1 game from hoopR-mbb-raw game JSONs.
Writes one compact JSON line per game; resumable."""
import sys, json, os, urllib.request, pandas as pd, concurrent.futures as cf
d, out = sys.argv[1], sys.argv[2]; seasons = [int(x) for x in sys.argv[3].split(',')]
done = set()
if os.path.exists(out):
    for l in open(out):
        try: done.add(json.loads(l)['g'])
        except Exception: pass
def strip(o):
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items() if k not in ('links', 'logos', 'team', 'link', 'header', 'footer')}
    if isinstance(o, list): return [strip(v) for v in o]
    return o
def get(g):
    for t in range(3):
        try:
            j = json.load(urllib.request.urlopen(f'https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-raw/main/mbb/json/final/{g}.json', timeout=90))
            return {'g': g, 'pc': strip(j.get('pickcenter') or []), 'odds': strip(j.get('odds') or []), 'pred': strip(j.get('predictor') or {})}
        except Exception as e: err = str(e)
    return {'g': g, 'err': err}
ids = []
for y in seasons:
    b = pd.read_parquet(f'{d}/team_box_{y}.parquet', columns=['game_id'])
    cnt = b.game_id.value_counts(); ids += [int(g) for g in cnt[cnt == 2].index if int(g) not in done]
print('to fetch', len(ids), flush=True)
with open(out, 'a') as f, cf.ThreadPoolExecutor(32) as ex:
    for i, r in enumerate(ex.map(get, ids)):
        f.write(json.dumps(r, separators=(',', ':')) + '\n')
        if i % 2000 == 0: print(i, flush=True); f.flush()
