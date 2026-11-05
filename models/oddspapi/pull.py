"""OddsPapi 2026 price history for US leagues (NFL, CFB, NBA, NHL, MLB): main lines + every full-game player prop.
Fixture lists: op/us/fx_<lg>.json (1 billable call per league: /fixtures?tournamentId=). History is free but 5 s apart
per key and ONE loop per key at a time (tennis thread shares the key; take turns).
  python3 pull.py hist LG[,LG..] MINUTES [FROM YYYY-MM-DD] [TO YYYY-MM-DD]   # resumable; newest first
  python3 pull.py retry LG[,LG..] MINUTES                                   # re-pull stalled fixtures, one book per call
  python3 pull.py names                                                      # 2 billable calls: id->name from upcoming odds
Writes op/us/hist_<lg>.jsonl: {fid, lg, start, home, away, books:{bk:{mid:{oid:{pid:[open, d24, close, n]}}}}}
open/d24/close = [seconds before start, decimal price, limit]; d24 = last price at least 24 h before start;
then n price changes and alive (1 = that line was still offered at the start)."""
import os, sys, json, time, datetime as dt, urllib.request, urllib.error
KEY = os.environ['ODDSPAPI_API_KEY']; BASE = 'https://api.oddspapi.io/v4'
OP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'op'); US = f'{OP}/us'
BOOKS = 'pinnacle,draftkings,fanduel'
CAP = 25
LG = {'nfl': 31, 'nba': 132, 'nhl': 234, 'mlb': 109, 'cfb': 27653, 'cbb': (648, 28370)}  # cbb = regular season + NCAA tournament
PERIODS = {'result', 'fulltime', 'fulltime+overtime'}
_M = None
def markets():
    global _M
    if _M is None:
        _M = {}
        for m in json.load(open(f'{OP}/markets.json')):
            if m['sportId'] in (11, 13, 14, 15) and m['period'] in PERIODS and (m['playerProp'] or m['marketType'] in ('spreads', 'totals', 'teamtotals-team1', 'teamtotals-team2') or m['marketName'].startswith('Winner')):
                _M[str(m['marketId'])] = m
    return _M

def get(path, **q):
    """curl with a hard cap per attempt: the API sometimes trickles a response for many minutes (seen 2026-10-28,
    erratic per fixture), which a socket read timeout never catches. A timed-out fixture is logged as an error and
    skipped; `retry` re-pulls those one book per call."""
    import subprocess
    q['apiKey'] = KEY
    url = BASE + path + '?' + '&'.join(f'{k}={v}' for k, v in q.items())
    for _ in range(2):
        r = subprocess.run(['curl', '-sS', '--max-time', str(CAP), '-w', '\n%{http_code}', url], capture_output=True, text=True)
        body, _, code = r.stdout.rpartition('\n')
        if code == '200':
            try: return json.loads(body)
            except ValueError: pass
        elif code == '429': time.sleep(5.5); continue
        elif code not in ('000', ''): return {'_error': int(code), 'body': body[:300]}
        if 'timed out' in r.stderr: return {'_error': 'timeout'}
        print('net', code, r.stderr[:100].strip(), flush=True); time.sleep(6)
    return {'_error': 'net'}

LOCK = os.path.normpath(f'{OP}/../app/data/odds/oddspapi.lock'); OWNER = 'models/oddspapi/pull.py'
# Bet Desk OddsPapi pulls (J 2026-10-28, America/Denver; app/routines/odds_pulls.md): (weekday or None, start, minutes held)
SWEEPS = [(None, '06:25', 150), (None, '11:55', 45), (None, '22:10', 150), (6, '20:40', 30), (1, '15:25', 30)]
def in_sweep_window():
    import zoneinfo
    now = dt.datetime.now(zoneinfo.ZoneInfo('America/Denver'))
    for wd, hhmm, mins in SWEEPS:
        h, m = map(int, hhmm.split(':')); st = now.replace(hour=h, minute=m, second=0, microsecond=0)
        for d in (0, -1):   # a window that started yesterday evening can run past midnight
            s0 = st + dt.timedelta(days=d)
            if (wd is None or s0.weekday() == wd) and s0 <= now < s0 + dt.timedelta(minutes=mins): return True
    return False

def lock():
    """Shared-key etiquette (op_live.py): wait while another owner holds an unexpired lock, then hold it ~10 min at a time.
    Also sit out the Bet Desk sweep windows (SWEEPS) even without a lock,
    and re-read after writing so a sweep that wrote its lock at the same moment wins the race."""
    while True:
        if in_sweep_window(): time.sleep(60); continue
        try: L = json.load(open(LOCK))
        except Exception: L = {}
        if L.get('owner') not in (None, OWNER) and ts(L.get('until', '1970-01-01T00:00:00Z')) > time.time(): time.sleep(30); continue
        json.dump({'owner': OWNER, 'until': dt.datetime.utcfromtimestamp(time.time() + 600).strftime('%Y-%m-%dT%H:%M:%SZ')}, open(LOCK, 'w'))
        time.sleep(1)
        try:
            if json.load(open(LOCK)).get('owner') == OWNER: return
        except Exception: pass
def unlock():
    try:
        if json.load(open(LOCK)).get('owner') == OWNER: os.remove(LOCK)
    except Exception: pass

ts = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()

def compact(f, lg, h):
    st = ts(f['startTime']); M = markets()
    rec = {'fid': f['fixtureId'], 'lg': lg, 'start': f['startTime'], 'home': f['participant1Name'], 'away': f['participant2Name'], 'books': {}}
    for bk, v in (h.get('bookmakers') or {}).items():
        bm = {}
        for mid, m in v.get('markets', {}).items():
            if mid not in M: continue
            outs = {}
            for oid, o in m['outcomes'].items():
                pl = {}
                for pid, hl in o['players'].items():
                    pre = sorted((x for x in hl if ts(x['createdAt']) < st), key=lambda x: x['createdAt'])
                    if not pre: continue
                    pt = lambda x: [round(st - ts(x['createdAt'])), x['price'], x.get('limit')]
                    d24 = [x for x in pre if st - ts(x['createdAt']) >= 86400]
                    act = [x for x in pre if x.get('active', True)] or pre
                    # close = last ACTIVE price before start; alive = the line was still up at the start (not pulled/moved off)
                    pl[pid] = [pt(pre[0]), pt(d24[-1]) if d24 else None, pt(act[-1]), len(pre), int(bool(pre[-1].get('active', True)))]
                if pl: outs[oid] = pl
            if outs: bm[mid] = outs
        rec['books'][bk] = bm
    return rec

def hist(lgs, minutes, lo='2026-01-01', hi=None):
    end = time.time() + minutes * 60
    hi = hi or (dt.datetime.utcnow() - dt.timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M')
    todo = []
    for lg in lgs:
        path = f'{US}/hist_{lg}.jsonl'; done = set()
        if os.path.exists(path):
            for line in open(path):
                try: done.add(json.loads(line)['fid'])
                except Exception: pass
        F = [f for f in json.load(open(f'{US}/fx_{lg}.json')) if lo <= f['startTime'] < hi and f.get('statusId') == 2 and f['fixtureId'] not in done]
        todo += [(lg, f) for f in F]
    todo.sort(key=lambda x: x[1]['startTime'], reverse=True)
    print('todo', len(todo), flush=True); n = 0
    for lg, f in todo:
        if time.time() > end: break
        lock(); t0 = time.time()
        h = get('/historical-odds', fixtureId=f['fixtureId'], bookmakers=BOOKS)
        with open(f'{US}/hist_{lg}.jsonl', 'a') as out:
            out.write(json.dumps({'fid': f['fixtureId'], 'lg': lg, 'err': h} if '_error' in h else compact(f, lg, h), separators=(',', ':')) + '\n')
        n += 1
        if time.time() - t0 > 15: print('slow', round(time.time() - t0), f['fixtureId'], flush=True)
        if n % 50 == 0: print(n, lg, f['startTime'], flush=True)
        time.sleep(max(0, 5.2 - (time.time() - t0)))
    unlock(); print('pulled', n, flush=True)

def retry(lgs, minutes):
    """Re-pull fixtures logged as timeouts/network errors (404 = no history, skipped), one book per call, and rewrite them in place."""
    end = time.time() + minutes * 60
    for lg in lgs:
        path = f'{US}/hist_{lg}.jsonl'; rows = []
        for line in open(path):
            try: rows.append(json.loads(line))
            except ValueError: pass
        F = {f['fixtureId']: f for f in json.load(open(f'{US}/fx_{lg}.json'))}
        n = 0
        for i, r in enumerate(rows):
            if 'err' not in r or str(r['err']).find('404') >= 0 or r['fid'] not in F or time.time() > end: continue
            books = {}
            for bk in BOOKS.split(','):
                lock(); t0 = time.time()
                h = get('/historical-odds', fixtureId=r['fid'], bookmakers=bk)
                if '_error' not in h: books.update(compact(F[r['fid']], lg, h)['books'])
                time.sleep(max(0, 5.2 - (time.time() - t0)))
            if books:
                rows[i] = compact(F[r['fid']], lg, {}); rows[i]['books'] = books; n += 1
        with open(path + '.tmp', 'w') as out:
            for r in rows: out.write(json.dumps(r, separators=(',', ':')) + '\n')
        os.replace(path + '.tmp', path); print(lg, 'retried ok', n, flush=True)
    unlock()

def names():
    path = f'{US}/player_ids.json'; N = json.load(open(path)) if os.path.exists(path) else {}
    for b in ('draftkings', 'fanduel'):
        d = get('/odds-by-tournaments', bookmaker=b, tournamentIds=','.join(str(t) for v in LG.values() for t in (v if isinstance(v, tuple) else (v,))))
        for fx in (d if isinstance(d, list) else []):
            for v in fx.get('bookmakerOdds', {}).values():
                for m in v.get('markets', {}).values():
                    for o in m['outcomes'].values():
                        for pid, p in o['players'].items():
                            if p.get('playerName'): N[pid] = p['playerName']
        time.sleep(2)
    json.dump(N, open(path, 'w')); print(len(N), 'player ids named')

if __name__ == '__main__':
    if sys.argv[1] == 'names': names()
    elif sys.argv[1] == 'retry': retry(sys.argv[2].split(','), float(sys.argv[3]))
    else: hist(sys.argv[2].split(','), float(sys.argv[3]), *(sys.argv[4:6]))
