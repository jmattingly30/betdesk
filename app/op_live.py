"""Free near-live game lines from OddsPapi (J 2026-10-27: split pulls between The Odds API and OddsPapi, no paid feed).

python3 op_live.py [NFL CFB NBA CBB NHL] # default: all five; writes data/odds/{SPORT}.json in The Odds API's format

OddsPapi's /historical-odds is free (not billed) and returns every price change up to the last few minutes, including
for games that have not started, so it works as a live feed: 3 books per call, 5 s apart per key (shared with other
threads: see LOCK below). Only the fixture lists are billed (250 calls a month): one call per sport family covers 9 days
and is cached 24 hours (football = NFL + CFB, basketball = NBA + CBB), so this costs at most 2 billed calls a day.
The current main line per book = the active handicap/total whose two prices are closest to even. Output matches the
Odds API reply (home_team, away_team, bookmakers[].markets h2h/spreads/totals, American odds) so export_slates.py and
line_tools.py read it unchanged; export_slates.py uses it instead of spending Odds API credits when it is under 6 h old.
Books: J's Colorado books + Pinnacle and BetOnline for the fair price (4 calls per game)."""
import json, os, sys, time, gzip, http.client, urllib.request, urllib.error, datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__)); D = f'{HERE}/data'; OP = f'{D}/odds/op'
BASE = 'https://api.oddspapi.io/v4'; KEY = os.environ.get('ODDSPAPI_API_KEY')
NOW = dt.datetime.now(dt.timezone.utc)
# 2026-11-07: the Saturday CFB sweep ran 1h50m and the 2-hour background limit killed the whole run before CBB/NBA/NHL.
# OP_DEADLINE_MIN (minutes from start): a sport still sweeping at the deadline stops after its current game, keeps the
# games it swept (earliest kickoffs first) plus the previous file's lines for the rest, and later sports are skipped.
T0, DEADLINE = time.time(), float(os.environ.get('OP_DEADLINE_MIN') or 0) * 60
late = lambda: DEADLINE and time.time() - T0 > DEADLINE
GROUPS = [['pinnacle', 'draftkings', 'fanduel'], ['betmgm', 'caesars', 'espnbet'], ['fanatics', 'betrivers', 'bet365'],
          ['ballybet', 'circasports', 'betonline.ag']]
# J 2026-10-29: noon and night pulls run OP_GROUPS=3 (skip Bally, Circa, BetOnline; the morning pull keeps all 12 books).
# The skipped books keep their last prices from the previous pull's file, with that pull's last_update.
NGROUPS = int(os.environ.get('OP_GROUPS', len(GROUPS)))
OUR_KEY = {'caesars': 'williamhill_us', 'betonline.ag': 'betonlineag'}      # Odds API book keys line_tools/export_slates use
SPORTS = dict(NFL=(14, 'NFL', 9), CFB=(14, 'NCAA, Regular Season', 7), NBA=(11, 'NBA', 2), CBB=(11, 'NCAA, Regular Season', 2),
              NHL=(15, 'NHL', 2))  # sportId, tournament, days ahead
# J 2026-10-28: Colorado books post no college player props, so CFB/CBB sweeps keep game lines only; every pro sport keeps props
PROP_SPORTS = {'NFL', 'NBA', 'NHL'}
MODEL_PROPS = {'CFB'}   # J 2026-10-29: college props kept as CFB game-model inputs only (write_model_props), never as bets
FIX_HOURS = 48          # fixture lists cover 9 days; refresh every 2 days (billed)
NAME_BOOKS = ['draftkings', 'bet365', 'fanduel', 'betrivers']  # billed name lookups, in order, only while 5+ players are unnamed
NAMES = {'Winner (incl. overtime)': 'h2h', 'Handicap (incl. overtime)': 'spreads', 'Total (incl. overtime)': 'totals',
         # hockey names its full-game markets with shootouts included (J 2026-10-28: NHL game lines)
         'Winner (incl. overtime and penalties)': 'h2h', 'Handicap (incl. overtime and penalties)': 'spreads',
         'Total (incl. overtime and penalties)': 'totals',
         # basketball names its full-game total 'Over Under' (J 2026-11-02: NBA + CBB totals were missing from every sweep)
         'Over Under (incl. overtime)': 'totals'}
GAP = 5.3
# One OddsPapi loop per key at a time: a live sweep writes LOCK {owner, until}; history loops in other threads (tennis
# op_pull.py, odds-history thread) pause while another owner's lock is unexpired.
LOCK = f'{D}/odds/oddspapi.lock'

def call(path, **q):
    q['apiKey'] = KEY; url = f'{BASE}{path}?' + '&'.join(f'{k}={v}' for k, v in q.items())
    for i in range(60):   # backs off up to ~6 min in all while another loop finishes with the key
        try:
            with urllib.request.urlopen(url, timeout=120) as r: b = r.read()
            if b'rate limited' in b[:300]: time.sleep(min(GAP * (1 + i // 5), 20)); continue
            return json.loads(b)
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(min(GAP * (1 + i // 5), 20)); continue
            if e.code == 404: return {}   # no prices at these books for this fixture
            if e.code >= 500 and i < 3: time.sleep(GAP * 2); continue   # Cloudflare 520 killed the first tennis sweep 2026-10-29
            raise
        except (ValueError, OSError, http.client.HTTPException) as e:   # cut-off or stalled reply: one bad game killed the CFB/NHL sweeps 2026-10-28
            if i < 3: time.sleep(GAP); continue
            print('skipping', q.get('fixtureId'), 'after 4 bad replies:', type(e).__name__); return {}
    raise RuntimeError(f'{path}: rate limited 60 times (another loop on the key?)')

def lock(hold_min, first=True):
    """Live sweeps take priority: write our lock even over a history loop's, then (first time) give that loop ~3 min to
    see it and pause (tennis thread's op_pull.py checks every 20 calls). Re-written after every game so it stays ours."""
    other = None
    if first and os.path.exists(LOCK):
        try:
            L = json.load(open(LOCK))
            if L.get('owner') != 'op_live' and dt.datetime.fromisoformat(L['until']) > dt.datetime.now(dt.timezone.utc): other = L.get('owner')
        except Exception: pass
    json.dump(dict(owner='op_live', until=(dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=hold_min)).isoformat()), open(LOCK, 'w'))
    if other: print('history loop', other, 'running; waiting 3 min for it to pause'); time.sleep(185)

def unlock():
    try:
        if json.load(open(LOCK)).get('owner') == 'op_live': os.remove(LOCK)
    except Exception: pass

def fixtures(sid):
    """Next 9 days of fixtures for a sport family, cached FIX_HOURS (1 billed call)."""
    f = f'{OP}/fixtures_{sid}.json'
    if os.path.exists(f):
        c = json.load(open(f))
        if NOW - dt.datetime.fromisoformat(c['made']) < dt.timedelta(hours=20 if sid == 12 else FIX_HOURS): return c['fx']   # tennis draws fill daily
    fx = call('/fixtures', sportId=sid, **{'from': NOW.strftime('%Y-%m-%dT%H:%M:%SZ'), 'to': (NOW + dt.timedelta(days=9)).strftime('%Y-%m-%dT%H:%M:%SZ')})
    if not isinstance(fx, list): return []   # bad reply: don't cache an empty list for 2 days
    os.makedirs(OP, exist_ok=True); json.dump(dict(made=NOW.isoformat(), fx=fx), open(f, 'w'), separators=(',', ':'))
    print('billed: fixtures sport', sid, len(fx))
    return fx

def american(d): return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))

def main_lines(book, M, p1, p2, sp=None):
    """Current main h2h / spread / total for one book: {'h2h': [...], 'spreads': [...], 'totals': [...]} in Odds API outcome form."""
    cand = {'h2h': [], 'spreads': [], 'totals': []}
    for mid, m in book.get('markets', {}).items():
        info = M.get(mid)
        if not info or info[1] not in NAMES: continue
        kind, hcap, names = NAMES[info[1]], info[2], info[5]; pr = {}; t0 = []
        if sp == 'NHL' and kind == 'spreads' and abs(hcap) != 1.5: continue   # the NHL main spread is the 1.5 puck line, never near even
        for oid, o in m.get('outcomes', {}).items():
            h = (o.get('players') or {}).get('0') or []
            if h and h[-1].get('active') and h[-1].get('price', 0) > 1: pr[names.get(oid, oid)] = h[-1]['price']; t0.append(min((x.get('createdAt') or '9') for x in h))
        if len(pr) == 2: cand[kind].append((hcap, pr, (min(t0) if min(t0) != '9' else None)))
    out = []
    for kind, C in cand.items():
        # a two-way rung whose prices sum under 100% implied (DK 113.5 +230/+245 in the 2026-11-02 NBA total feed) is a stale or
        # mislabelled market, never the main line: it would win the lowest-hold test below and create fake arbs
        C = [c for c in C if sum(1 / v for v in c[1].values()) >= 1.0]
        if not C: continue
        # main line = the lowest-hold rung near even money (alt rungs carry more hold; the closest-to-even rung alone can
        # be an alt when the main is shaded, e.g. DK 47.5 -122/-109 vs 48.5 -105/-115 after a move)
        def score(c):
            a, b = (1 / v for v in c[1].values()); return (abs(a - b) > 0.12, a + b - 1 + 0.25 * abs(a - b))
        hcap, pr, posted = min(C, key=score)
        if kind == 'h2h': oc = [dict(name=p1, price=american(pr['1'])), dict(name=p2, price=american(pr['2']))] if {'1', '2'} <= set(pr) else None
        elif kind == 'spreads':   # handicap is participant 1's (home) line
            oc = [dict(name=p1, price=american(pr['1']), point=hcap), dict(name=p2, price=american(pr['2']), point=-hcap)] if {'1', '2'} <= set(pr) else None
        else: oc = [dict(name='Over', price=american(pr['Over']), point=hcap), dict(name='Under', price=american(pr['Under']), point=hcap)] if {'Over', 'Under'} <= set(pr) else None
        # posted (J 2026-10-28 "look ahead"): when this book first put up this line, from OddsPapi's own price history
        if oc: out.append(dict(key=kind, outcomes=oc, posted=posted))
    return out

def props(book, M):
    """Every active player prop at one book: [(playerId, stat, line, {side: American})]. Over/Under and Yes/No markets."""
    out = []
    for mid, m in book.get('markets', {}).items():
        info = M.get(mid)
        if not info or not info[4]: continue
        stat = info[1].replace('Over Under Player ', '').replace('Player ', '').replace(' (incl. overtime)', '').strip()
        by, t0 = {}, {}
        for oid, o in m.get('outcomes', {}).items():
            for pid, h in (o.get('players') or {}).items():
                if pid == '0' or not h or not h[-1].get('active') or h[-1].get('price', 0) <= 1: continue
                by.setdefault(pid, {})[info[5].get(oid, oid)] = american(h[-1]['price'])
                c = min((x.get('createdAt') or '9') for x in h)
                if c != '9': t0[pid] = min(t0.get(pid, c), c)
        out += [(pid, stat + (' ladder' if any(k.endswith('+') for k in sd) else ''), info[2], sd, t0.get(pid)) for pid, sd in by.items()]   # ladder = 250+, 275+ ... alt lines
    return out

NAMES_F = f'{OP}/players.json'
def names_for(pids, sid, tid):
    """playerId -> 'First Last'. The free history has ids only; names come from the billed /odds-by-tournaments (one book
    per call, whole league). Books are tried in NAME_BOOKS order while 5+ ids are still unknown, each at most once a
    league a day (DraftKings only names players it has posted, so bet365-only players need bet365's call)."""
    N = json.load(open(NAMES_F)) if os.path.exists(NAMES_F) else {'ids': {}, 'asked': {}}
    for book in NAME_BOOKS:
        miss = [p for p in pids if p not in N['ids']]
        if len(miss) < 5: break
        key = f'{tid}:{book}'; last = N['asked'].get(key)
        if last and NOW - dt.datetime.fromisoformat(last) < dt.timedelta(hours=20): continue
        d = call('/odds-by-tournaments', tournamentIds=tid, bookmaker=book); print('billed: player names, tournament', tid, book)
        for g in d if isinstance(d, list) else []:
            for bk in (g.get('bookmakerOdds') or {}).values():
                for m in (bk.get('markets') or {}).values():
                    for o in (m.get('outcomes') or {}).values():
                        for pid, p in (o.get('players') or {}).items():
                            n = p.get('playerName') if isinstance(p, dict) else None
                            if pid != '0' and n: N['ids'][pid] = ' '.join(reversed([x.strip() for x in n.split(',', 1)])) if ',' in n else n
        N['asked'][key] = NOW.isoformat(); json.dump(N, open(NAMES_F, 'w'))
        time.sleep(GAP)
    return N['ids']

def sweep(sp, M):
    sid, tour, days = SPORTS[sp]
    fx = [f for f in fixtures(sid) if f.get('tournamentName') == tour and f.get('hasOdds') and f.get('statusId') == 0
          and NOW < dt.datetime.fromisoformat(f['startTime'].replace('Z', '+00:00')) < NOW + dt.timedelta(days=days)]
    ev, P, cut = [], [], None
    if late(): print(sp, 'skipped: OP_DEADLINE_MIN reached before it started; leaving its files'); return
    for n, f in enumerate(sorted(fx, key=lambda f: f['startTime'])):
        if late(): cut = n; print(sp, f'deadline: stopped after {n} of {len(fx)} games (earliest first); the rest keep the previous file\'s lines'); break
        pg = dict(id=f['fixtureId'], k=f['startTime'].replace('.000Z', 'Z'), h=f['participant1Name'], a=f['participant2Name'], tid=f['tournamentId'], p={}, t={})
        e = dict(id=f['fixtureId'], sport_key=sp, commence_time=f['startTime'].replace('.000Z', 'Z'), home_team=f['participant1Name'],
                 away_team=f['participant2Name'], bookmakers=[], src='oddspapi')
        for gi, g in enumerate(GROUPS[:NGROUPS]):
            h = call('/historical-odds', fixtureId=f['fixtureId'], bookmakers=','.join(g)); time.sleep(GAP)
            # Games Pinnacle/DK/FD don't price (FCS/D2/Ivy: no Colorado book prices them either, J 2026-10-28) cost 1 call, not 4
            if gi == 0 and not h.get('bookmakers'): break
            for b, v in (h.get('bookmakers') or {}).items():
                for pid, stat, pt, sd, posted in (props(v, M) if sp in PROP_SPORTS or sp in MODEL_PROPS else []):
                    pg['p'].setdefault(pid, {}).setdefault(stat, {}).setdefault(str(pt), {})[OUR_KEY.get(b, b)] = sd
                    if posted: pg['t'].setdefault(pid, {}).setdefault(stat, {}).setdefault(str(pt), {})[OUR_KEY.get(b, b)] = posted
                mk = main_lines(v, M, e['home_team'], e['away_team'], sp)
                if mk: e['bookmakers'].append(dict(key=OUR_KEY.get(b, b), title=b, last_update=NOW.isoformat(), markets=mk))
        if e['bookmakers']: ev.append(e)
        if pg['p']: P.append(pg)
        lock(30, first=False)
    for g in P: drop_placeholders(g['p'])
    if sp in MODEL_PROPS: write_model_props(sp, P)
    else: write_props(sp, sid, P, keep_old=cut is not None)
    if not ev: print(sp, 'no OddsPapi games; leaving', f'data/odds/{sp}.json'); return   # an empty file would block the Odds API fallback
    carry_books(sp, ev)
    if cut is not None:   # stopped early: keep the previous file's events for the games not reached
        try: old = json.load(open(f'{D}/odds/{sp}.json')).get('events', [])
        except (OSError, ValueError): old = []
        have = {e['id'] for e in ev}
        ev += [e for e in old if e['id'] not in have and e.get('commence_time', '') > NOW.isoformat()[:19]]
    json.dump(dict(made=NOW.isoformat(), src='oddspapi', left=None, used=None, events=ev), open(f'{D}/odds/{sp}.json', 'w'), separators=(',', ':'))
    if sp == 'NHL':   # J 2026-10-28: keep every day's NHL lines (no NHL closing-line history yet; desk has no NHL game card until the model is built)
        hf = f'{D}/odds/nhl_line_history.json.gz'
        H = json.load(gzip.open(hf, 'rt')) if os.path.exists(hf) else {}
        H[NOW.isoformat(timespec='minutes')] = ev
        json.dump(H, gzip.open(hf, 'wt'), separators=(',', ':'))
    print(sp, len(ev), 'games from OddsPapi,', len(fx) * NGROUPS, 'free calls')

def skipped():
    return {OUR_KEY.get(b, b) for g in GROUPS[NGROUPS:] for b in g}

def carry_books(sp, ev):
    """With OP_GROUPS < 4, copy the skipped books' game lines from the previous {SPORT}.json into this pull's events."""
    sk, f = skipped(), f'{D}/odds/{sp}.json'
    if not sk or not ev or not os.path.exists(f): return
    try: old = {e['id']: e for e in json.load(open(f)).get('events', [])}
    except (ValueError, OSError): return
    for e in ev:
        have = {b['key'] for b in e['bookmakers']}
        e['bookmakers'] += [b for b in old.get(e['id'], {}).get('bookmakers', []) if b['key'] in sk and b['key'] not in have]

def carry_props(sp, games):
    """Same for player props: skipped books' prices from the previous props_{SPORT}.json, matched by game, player, stat, line."""
    sk, f = skipped(), f'{D}/odds/props_{sp}.json'
    if not sk or not os.path.exists(f): return
    try: old = {g['id']: g for g in json.load(open(f)).get('games', [])}
    except (ValueError, OSError): return
    for g in games:
        og = old.get(g['id'], {}).get('players', {})
        for name, st in og.items():
            for stat, lines in st.items():
                for line, bs in lines.items():
                    keep = {b: v for b, v in bs.items() if b in sk}
                    if keep: g['players'].setdefault(name, {}).setdefault(stat, {}).setdefault(line, {}).update({b: v for b, v in keep.items() if b not in g['players'][name][stat][line]})

def drop_placeholders(pp):
    """bet365 posts -120/-120 on several lines for one player and stat (e.g. 184.5 and 219.5): placeholders, not prices,
    and they make fake middles. Drop a book's two-sided -120/-120 rows when that player-stat has 2+ of them, and every
    bet365 -120/-120 row (its default placeholder price on props, desk thread 2026-10-27)."""
    for st in pp.values():
        for stat, lines in st.items():
            for b in {b for l in lines.values() for b in l}:
                ph = [l for l, bs in lines.items() if bs.get(b, {}).get('Over') == -120 and bs.get(b, {}).get('Under') == -120]
                if len(ph) >= 2 or (b == 'bet365' and ph):
                    for l in ph: del lines[l][b]
            for l in [l for l, bs in lines.items() if not bs]: del lines[l]

def write_model_props(sp, P):
    """data/odds/model_props_{SPORT}.json: college props as GAME-model inputs only (J 2026-10-29: usage priors + missing-star
    flag for CFB; Colorado books post no college props, so they are never shown as bets). Player ids only: no billed
    name lookups (models/oddspapi/cfb_usage.py maps ids to teams from their past games)."""
    if not P: return
    json.dump(dict(made=NOW.isoformat(), src='oddspapi', games=[dict(id=g['id'], k=g['k'], h=g['h'], a=g['a'], p=g['p']) for g in P]),
              open(f'{D}/odds/model_props_{sp}.json', 'w'), separators=(',', ':'))
    print(sp, 'model props:', sum(len(g['p']) for g in P), 'players in', len(P), 'games')

def write_props(sp, sid, P, keep_old=False):
    """data/odds/props_{SPORT}.json: {made, games: [{id, k, h, a, players: {name: {stat: {line: {book: {Over|Under|Yes|No: American}}}}}}]}"""
    if not P: return
    ids = {pid for g in P for pid in g['p']}; N = {}
    for tid in {g['tid'] for g in P}: N.update(names_for(ids, sid, tid))
    # posted: {player: {stat: {line: {book: ISO time the book first posted that line}}}} for the desk's "look ahead"
    games = [dict(id=g['id'], k=g['k'], h=g['h'], a=g['a'], players={N.get(pid, f'id{pid}'): st for pid, st in g['p'].items()},
                  posted={N.get(pid, f'id{pid}'): st for pid, st in g.get('t', {}).items()}) for g in P]
    carry_props(sp, games)
    if keep_old:   # deadline cut the sweep: games it did not reach keep the previous file's props
        try: games += [g for g in json.load(open(f'{D}/odds/props_{sp}.json')).get('games', []) if g['id'] not in {x['id'] for x in games} and g.get('k', '') > NOW.isoformat()[:19]]
        except (OSError, ValueError): pass
    json.dump(dict(made=NOW.isoformat(), src='oddspapi', games=games), open(f'{D}/odds/props_{sp}.json', 'w'), separators=(',', ':'))
    print(sp, 'props:', sum(len(g['players']) for g in games), 'players,', sum(1 for g in games for v in g['players'].values() for s in v.values() for _ in s), 'lines;',
          sum(1 for pid in ids if pid not in N), 'players without a name')

def rename(sp):
    """Fill names (and drop bet365 placeholders) in an existing props file without re-sweeping: python3 op_live.py --names NFL"""
    f = f'{D}/odds/props_{sp}.json'
    if not os.path.exists(f): return
    d = json.load(open(f)); sid, tour, _ = SPORTS[sp]
    tid = next((x['tournamentId'] for x in fixtures(sid) if x.get('tournamentName') == tour), None)
    ids = {n[2:] for g in d['games'] for n in g['players'] if n.startswith('id')}
    N = names_for(ids, sid, tid) if ids and tid else {}
    for g in d['games']:
        g['players'] = {(N.get(n[2:], n) if n.startswith('id') else n): v for n, v in g['players'].items()}
        drop_placeholders(g['players'])
    json.dump(d, open(f, 'w'), separators=(',', ':'))
    print(sp, 'renamed', sum(1 for i in ids if i in N), 'of', len(ids), 'unnamed players')

# Tennis (tennis thread spec 2026-10-28; J: morning + night pulls only, once the Bet Desk rebuild has tennis). Moneyline
# (market 121) for singles in these categories, next 36 h. "open" is the first price the book ever posted (from the
# price history, else the first price we saw, carried forward and never overwritten): the edge is against openers.
TENNIS_CATS = {'ITF Women', 'Challenger', 'WTA', 'WTA 125K', 'ATP'}
TENNIS_GROUPS = [['pinnacle', 'bet365', 'draftkings'], ['fanduel', 'betmgm', 'caesars'], ['espnbet', 'fanatics', 'betrivers']]

def sweep_tennis():
    out_f = f'{D}/odds/tennis.json'
    old = {r['fid']: r for r in json.load(open(out_f))} if os.path.exists(out_f) else {}
    fx = [f for f in fixtures(12) if f.get('categoryName') in TENNIS_CATS and 'Doubles' not in (f.get('tournamentName') or '')
          and '/' not in (f.get('participant1Name') or '') + (f.get('participant2Name') or '') and f.get('statusId') in (0, None)
          and NOW < dt.datetime.fromisoformat(f['startTime'].replace('Z', '+00:00')) < NOW + dt.timedelta(hours=36)]
    rows = []
    for f in sorted(fx, key=lambda f: f['startTime']):
        r = dict(fid=f['fixtureId'], start=f['startTime'].replace('.000Z', 'Z'), cat=f['categoryName'], tournament=f['tournamentName'],
                 p1=f['participant1Name'], p2=f['participant2Name'], books=dict(old.get(f['fixtureId'], {}).get('books', {})))
        for gi, g in enumerate(TENNIS_GROUPS):
            h = call('/historical-odds', fixtureId=f['fixtureId'], bookmakers=','.join(g)); time.sleep(GAP)
            if gi == 0 and not h.get('bookmakers'): break    # nobody prices it: 1 call, not 3
            for b, v in (h.get('bookmakers') or {}).items():
                o = (v.get('markets') or {}).get('121', {}).get('outcomes', {})
                h1 = (o.get('121', {}).get('players') or {}).get('0') or []; h2 = (o.get('122', {}).get('players') or {}).get('0') or []
                h1 = sorted((x for x in h1 if x.get('price', 0) > 1), key=lambda x: x['createdAt']); h2 = sorted((x for x in h2 if x.get('price', 0) > 1), key=lambda x: x['createdAt'])
                if not h1 or not h2: continue
                k = OUR_KEY.get(b, b); prev = r['books'].get(k, {})
                first = [min(h1[0]['createdAt'], h2[0]['createdAt']), h1[0]['price'], h2[0]['price']]
                now = [max(h1[-1]['createdAt'], h2[-1]['createdAt']), h1[-1]['price'], h2[-1]['price']] if h1[-1].get('active', True) and h2[-1].get('active', True) else None
                op = prev.get('open') if prev.get('open') and prev['open'][0] <= first[0] else first   # never overwrite an earlier open
                r['books'][k] = dict(open=op, now=now)
        if r['books']: rows.append(r)
        lock(30, first=False)
    if not rows: print('TENNIS no priced matches; leaving', out_f); return
    # J 2026-11-06: matches that started keep their last pre-match prices so the desk can list the whole day (models/model-b/pregame.py)
    sys.path.insert(0, f'{HERE}/../models/model-b'); from pregame import keep
    rows = keep(out_f, rows, kick=lambda i, r: r.get('start'), key='fid', field=None)
    json.dump(rows, open(out_f, 'w'), separators=(',', ':'))
    print('TENNIS', len(rows), 'matches priced of', len(fx), '| by category', {c: sum(r['cat'] == c for r in rows) for c in TENNIS_CATS})

if __name__ == '__main__':
    if not KEY: sys.exit('ODDSPAPI_API_KEY not set')
    if '--names' in sys.argv:
        for sp in [s for s in sys.argv[1:] if s in SPORTS]: rename(sp)
        sys.exit()
    M = json.load(gzip.open(f'{D}/op_markets.json.gz', 'rt'))   # marketId -> [sportId, name, handicap, period, prop, {outcomeId: name}]
    todo = [s for s in sys.argv[1:] if s in SPORTS or s == 'TENNIS'] or list(SPORTS) + ['TENNIS']   # tennis in the default (morning + night pulls) since the 2026-10-30 desk redesign
    # 2026-11-05: several pull sessions resumed at once after a usage-limit pause; two live sweeps on one key collide.
    # A live lock held by another op_live run means a sweep is already running: stop (OP_FORCE=1 overrides).
    try:
        L = json.load(open(LOCK))
        if L.get('owner') == 'op_live' and dt.datetime.fromisoformat(L['until']) > dt.datetime.now(dt.timezone.utc) and not os.environ.get('OP_FORCE'):
            sys.exit(f"another op_live sweep holds the key until {L['until'][:16]} UTC; not starting a second one (OP_FORCE=1 to override)")
    except (FileNotFoundError, ValueError, KeyError): pass
    lock(30)   # re-written after every game, so a killed run frees the key within 30 min
    try:
        for sp in todo:
            try: sweep_tennis() if sp == 'TENNIS' else sweep(sp, M)
            except Exception as ex: print(sp, 'failed:', ex)
    finally: unlock()
