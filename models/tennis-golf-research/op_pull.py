"""OddsPapi pull for 2026 tennis: fixture lists (billable, 1 call per 9-day window) then free price histories.
Usage: op_pull.py fixtures | op_pull.py hist MAXN [CATS]
Writes ../op/fixtures_2026.json and appends compact records to ../op/hist.jsonl (resumable)."""
import os, sys, json, time, datetime as dt, urllib.request, urllib.error

KEY = os.environ['ODDSPAPI_API_KEY']
BASE = 'https://api.oddspapi.io/v4'
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'op')
MK = json.load(open(os.path.join(OUT, 'tennis_markets.json')))
BOOKS = os.environ.get('OP_BOOKS', 'pinnacle,bet365,draftkings')


def get(path, **q):
    """One API call with a hard 25 s cap (curl --max-time); a slow trickle that a socket timeout misses is cut off."""
    import subprocess
    q['apiKey'] = KEY
    url = BASE + path + '?' + '&'.join(f'{k}={v}' for k, v in q.items())
    for attempt in range(4):
        r = subprocess.run(['curl', '-sS', '--max-time', '25', '-w', '\n%{http_code}', url], capture_output=True, text=True)
        if r.returncode != 0:
            time.sleep(5); continue
        body, _, code = r.stdout.rpartition('\n')
        if code == '200':
            try:
                return json.loads(body)
            except Exception:
                time.sleep(5); continue
        if code == '429' and 'RATE' in body.upper():
            time.sleep(6); continue
        return {'_error': int(code) if code.isdigit() else code, 'body': body[:300]}
    return {'_error': 'net'}


def fixtures():
    allf = {}
    d = dt.datetime(2026, 1, 1)
    end = dt.datetime(2026, 10, 27)
    while d < end:
        e = min(d + dt.timedelta(days=9), end)
        r = get('/fixtures', sportId=12, **{'from': d.strftime('%Y-%m-%dT%H:%M:%SZ'), 'to': e.strftime('%Y-%m-%dT%H:%M:%SZ')})
        if isinstance(r, list):
            for f in r:
                allf[f['fixtureId']] = f
        print(d.date(), len(r) if isinstance(r, list) else r, flush=True)
        d = e
        time.sleep(1.5)
    json.dump(list(allf.values()), open(os.path.join(OUT, 'fixtures_2026.json'), 'w'))
    print('total', len(allf))


def ts(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()


LOCK = '/mnt/project-files/sports-analytics/app/data/odds/oddspapi.lock'
OWNER = 'tennis-history'


SWEEPS = [(None, '06:25', 150), (None, '11:55', 45), (None, '22:10', 150), (6, '20:40', 30), (1, '15:25', 30)]  # Mountain time, same as models/oddspapi/pull.py


def in_sweep_window():
    import zoneinfo
    now = dt.datetime.now(zoneinfo.ZoneInfo('America/Denver'))
    for wd, hhmm, mins in SWEEPS:
        h, m = map(int, hhmm.split(':')); st = now.replace(hour=h, minute=m, second=0, microsecond=0)
        for d in (0, -1):
            s0 = st + dt.timedelta(days=d)
            if (wd is None or s0.weekday() == wd) and s0 <= now < s0 + dt.timedelta(minutes=mins): return True
    return False


def lock_wait_and_hold(minutes=3):
    """Shared key: wait while another owner's unexpired lock exists, then write our own lock {owner, until}."""
    while True:
        try:
            L = json.load(open(LOCK))
            until = dt.datetime.fromisoformat(str(L.get('until')).replace('Z', '+00:00'))
            if L.get('owner') != OWNER and until > dt.datetime.now(dt.timezone.utc):
                time.sleep(30); continue
        except Exception:
            pass
        break
    try:
        os.makedirs(os.path.dirname(LOCK), exist_ok=True)
        json.dump({'owner': OWNER, 'until': (dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=minutes)).isoformat()}, open(LOCK, 'w'))
    except Exception:
        pass


def lock_release():
    try:
        if json.load(open(LOCK)).get('owner') == OWNER: os.remove(LOCK)
    except Exception:
        pass


def compact(f, h):
    start = ts(f['startTime'])
    rec = {'fid': f['fixtureId'], 'start': f['startTime'], 'cat': f['categoryName'], 'tour': f['tournamentName'],
           'p1': f['participant1Name'], 'p2': f['participant2Name'], 'books': {}}
    for bk, v in (h.get('bookmakers') or {}).items():
        bm = {}
        for mid, m in v.get('markets', {}).items():
            meta = MK.get(mid)
            if not meta:
                continue
            keep_series = mid == '121'
            if not (keep_series or mid == '123' or (meta['period'] == 'result' and meta['type'] in ('totals-games', 'spreads-games'))):
                continue
            outs = {}
            for oid, o in m['outcomes'].items():
                hist = sorted(o['players'].get('0', []), key=lambda x: x['createdAt'])
                if not hist:
                    continue
                pre = [x for x in hist if ts(x['createdAt']) < start]
                name = meta['out'].get(oid, oid)
                item = {'open': [pre[0]['createdAt'], pre[0]['price']] if pre else None,
                        'close': [pre[-1]['createdAt'], pre[-1]['price'], pre[-1].get('limit')] if pre else None,
                        'last': [hist[-1]['createdAt'], hist[-1]['price']]}
                if keep_series:
                    item['series'] = [[round(ts(x['createdAt']) - start), x['price']] for x in pre][-400:]
                outs[name] = item
            bm[mid] = {'t': meta['type'], 'h': meta['hcp'], 'o': outs}
        rec['books'][bk] = bm
    return rec


def hist(maxn, cats):
    F = json.load(open(os.path.join(OUT, 'fixtures_2026.json')))
    done = set()
    path = os.path.join(OUT, os.environ.get('OP_HIST', 'hist.jsonl'))
    if os.path.exists(path):
        for line in open(path):
            try:
                done.add(json.loads(line)['fid'])
            except Exception:
                pass
    F = [f for f in F if f['categoryName'] in cats and (f.get('statusId') == 2 or (f['categoryName'].startswith('ITF') and f.get('statusId') is None)) and 'Doubles' not in f['tournamentName']
         and '/' not in f['participant1Name'] and f['fixtureId'] not in done and f['startTime'] < (dt.datetime.utcnow() - dt.timedelta(hours=8)).strftime('%Y-%m-%dT%H:%M:%S')]
    if os.environ.get('OP_ONLY'):
        keep = set(json.load(open(os.environ['OP_ONLY']))); F = [f for f in F if f['fixtureId'] in keep]
    F.sort(key=lambda f: f['startTime'], reverse=True)
    print('todo', len(F), flush=True)
    n = 0
    with open(path, 'a') as out:
        for f in F[:maxn]:
            if n % 20 == 0:
                while in_sweep_window():
                    time.sleep(60)  # Bet Desk sweep windows (shared key)
                lock_wait_and_hold()
            t0 = time.time()
            h = get('/historical-odds', fixtureId=f['fixtureId'], bookmakers=BOOKS)
            if '_error' in h:
                out.write(json.dumps({'fid': f['fixtureId'], 'err': h}) + '\n')
            else:
                out.write(json.dumps(compact(f, h)) + '\n')
            out.flush()
            n += 1
            if n % 50 == 0:
                print(n, f['startTime'], flush=True)
            time.sleep(max(0, 5.2 - (time.time() - t0)))
    lock_release()


if __name__ == '__main__':
    if sys.argv[1] == 'fixtures':
        fixtures()
    else:
        cats = sys.argv[3].split(',') if len(sys.argv) > 3 else ['ATP', 'WTA', 'Challenger', 'WTA 125K']
        hist(int(sys.argv[2]), cats)
