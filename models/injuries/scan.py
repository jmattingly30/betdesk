"""Injury news scan (J 2026-11-06: "scan for injury news in the morning pulls and change our projections").

python3 models/injuries/scan.py            (run from the repo root; ~4 small ESPN requests, free, no key)

Source: ESPN's public injuries feed (site.api.espn.com .../injuries), the same reports ESPN shows on its injury pages,
updated through the day from team reports and beat writers. NBA and NFL are used; ESPN's CFB feed is stale (entries from
2020) and its CBB feed is empty, so college sports get no injury input yet (manual entries still work).

What it changes
- NBA props (app/export_nba_props.py --games models/injuries/nba_games.csv): Out players are taken off the roster (their
  minutes and shots go to teammates through the model's usual absence features) and questionable players get a chance of
  playing (q column). The NBA team model numbers on the props page use the same outs.
- NBA Model A (models/rebuild/basketball/live_future_nba.py, used by models/nba_ab/live.py and the shadow run): Out and
  Doubtful players are dropped from tonight's projected lineup even if they played the last game.
- NFL props (app/export_nfl_props.py --overrides models/injuries/nfl_overrides.csv): ESPN Out / Doubtful / Injured
  Reserve players are added to the out list on top of the official nflverse report the model already reads; J's manual
  rows in models/nfl-player/overrides.csv (QB picks etc.) are kept as they are.

Reading a status (NBA)
- Out: out. Doubtful: plays 25%. Day-To-Day: read from ESPN's note when it is at most 3 days old: "ruled out / will not
  play / is out" -> out, "doubtful" 25%, "questionable / game-time" 50%, "probable" 90%; otherwise not used.
- STALE CHECK: an entry is ignored if the player has played a game after the entry's date (news overtaken by events).
Manual entries (models/injuries/manual.json) win over the feed and expire after 'until' (YYYY-MM-DD, Denver):
  {"NBA": [{"name": "Anthony Edwards", "team": "MIN", "status": "out" | "q" | "in", "p": 0.5, "until": "2026-11-10",
  "why": "..."}], "NFL": [...]};  "in" forces a player to count as playing even if the feed lists him.

Safety
- If ESPN fails, the last good scan (app/data/injuries.json) is reused when it is under 36 hours old; older than that, no
  injury input is used (the old behaviour) and the step says so. The scan itself never fails the chain.
- Every status change versus the previous scan is appended to models/injuries/log.jsonl and listed under 'changes' in
  app/data/injuries.json (the page can show it later).
"""
import os, re, json, sys, datetime, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(f'{HERE}/../..')
OUT_JSON, LOG = f'{ROOT}/app/data/injuries.json', f'{HERE}/log.jsonl'
MANUAL, NFL_MANUAL = f'{HERE}/manual.json', f'{ROOT}/models/nfl-player/overrides.csv'
NBA_CSV, NFL_CSV = f'{HERE}/nba_games.csv', f'{HERE}/nfl_overrides.csv'
API = 'https://site.api.espn.com/apis/site/v2/sports/{}/{}'
LEAGUES = {'NBA': 'basketball/nba', 'NFL': 'football/nfl'}
MAX_AGE_H, DTD_DAYS = 36, 3
NOW = datetime.datetime.now(datetime.timezone.utc)
TODAY = (NOW - datetime.timedelta(hours=7)).strftime('%Y-%m-%d')   # Denver date (close enough for expiry)

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'betdesk-injury-scan/1.0'})
    with urllib.request.urlopen(req, timeout=30) as r: return json.load(r)

def read_note(st, note):
    """(p_play, reason) for one entry, or (None, why_unused)."""
    s = (st or '').lower(); c = (note or '').lower()
    if s == 'out': return 0.0, 'Out'
    if s == 'doubtful': return 0.25, 'Doubtful'
    if s == 'questionable': return 0.5, 'Questionable'
    if s in ('injured reserve', 'suspension', 'physically unable to perform'): return 0.0, st
    if s == 'day-to-day':
        if re.search(r'ruled out|will not play|won\'t play|is out|will miss|out for', c): return 0.0, 'Day-to-day, note says out'
        if 'doubtful' in c: return 0.25, 'Day-to-day, note says doubtful'
        if re.search(r'questionable|game-time|uncertain', c): return 0.5, 'Day-to-day, note says questionable'
        if 'probable' in c: return 0.9, 'Day-to-day, note says probable'
        return None, 'Day-to-day, note unclear'
    return None, f'status {st}'

def scan_league(lg):
    teams = get(API.format(LEAGUES[lg], 'teams'))
    fix = {'WSH': 'WAS', 'LAR': 'LA'} if lg == 'NFL' else {}   # nflverse codes for the NFL props model
    abbr = {t['team']['id']: fix.get(t['team']['abbreviation'], t['team']['abbreviation']) for t in teams['sports'][0]['leagues'][0]['teams']}
    d = get(API.format(LEAGUES[lg], 'injuries'))
    rows = []
    for t in d.get('injuries', []):
        tm = abbr.get(str(t.get('id')))
        if not tm: continue
        for i in t.get('injuries', []):
            a = i.get('athlete', {}); href = next((l.get('href', '') for l in a.get('links', [])), '')
            m = re.search(r'/id/(\d+)', href)
            rows.append(dict(team=tm, n=a.get('displayName'), pid=int(m.group(1)) if m else None,
                             pos=(a.get('position') or {}).get('abbreviation'), st=i.get('status'), d=(i.get('date') or '')[:16],
                             ret=(i.get('details') or {}).get('returnDate'), c=(i.get('shortComment') or '')[:200]))
    if lg == 'NBA' and len(rows) < 10: raise RuntimeError(f'only {len(rows)} NBA entries (feed looks broken)')
    return rows

def last_played(lg):
    """{pid: last date played} so stale entries (player has played since the entry) are skipped."""
    if lg != 'NBA': return {}
    try:
        import pandas as pd
        pg = pd.read_parquet(f'{ROOT}/models/nba-player/data/player_games.parquet', columns=['pid', 'date', 'played'])
        return pg[pg.played].groupby('pid').date.max().dt.strftime('%Y-%m-%d').to_dict()
    except Exception as e:
        print('stale check unavailable:', e); return {}

def judge(lg, rows):
    lp = last_played(lg); used = []
    for r in rows:
        p, why = read_note(r['st'], r['c'])
        age_d = (NOW - datetime.datetime.fromisoformat(r['d'] + ':00+00:00')).days if r['d'] else 99
        if p is not None and r['st'] == 'Day-To-Day' and age_d > DTD_DAYS: p, why = None, 'Day-to-day note too old'
        if p is not None and r['pid'] and lp.get(r['pid'], '') > r['d'][:10]: p, why = None, f"played since ({lp[r['pid']]})"
        if lg == 'NFL' and p is not None and p > 0.25: p, why = None, why + ' (NFL: official report handles questionable)'
        r.update(p=p, why=why, src='ESPN'); used.append(r)
    return used

def manual(lg, rows):
    try: M = json.load(open(MANUAL)).get(lg, [])
    except FileNotFoundError: return rows
    for e in M:
        if e.get('until') and e['until'] < TODAY: print('manual entry expired:', e.get('name')); continue
        key = (e['team'], e['name'].lower())
        rows = [r for r in rows if (r['team'], (r['n'] or '').lower()) != key]
        p = {'out': 0.0, 'in': None, 'q': float(e.get('p', 0.5))}.get(e.get('status', 'out'), 0.0)
        rows.append(dict(team=e['team'], n=e['name'], pid=None, pos=None, st='manual ' + e.get('status', 'out'), d=TODAY,
                         ret=None, c=e.get('why', ''), p=p, why='manual (J)', src='manual', until=e.get('until')))
    return rows

def write_nba(rows):
    import csv
    by = {}
    for r in rows:
        if r['p'] is None: continue
        o, q = by.setdefault(r['team'], ([], []))
        (o.append(r['n']) if r['p'] <= 0.0 else q.append(f"{r['n']}:{r['p']}"))
    with open(NBA_CSV, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['team', 'spread', 'total', 'out', 'q'])
        for t in sorted(by): w.writerow([t, '', '', ';'.join(by[t][0]), ';'.join(by[t][1])])
    print('NBA overrides ->', NBA_CSV, {t: len(v[0]) for t, v in sorted(by.items())})

def write_nfl(rows):
    import pandas as pd
    try: man = pd.read_csv(NFL_MANUAL, dtype=str).fillna('')
    except FileNotFoundError: man = pd.DataFrame(columns=['team', 'out', 'add', 'qb', 'until'])
    for c in ('team', 'out', 'add', 'qb', 'until'):
        if c not in man.columns: man[c] = ''
    if len(man): man = man[(man.until == '') | (man.until >= TODAY)]   # expired manual rows drop here too
    outs = {}
    for r in rows:
        if r['p'] is not None and r['p'] <= 0.25: outs.setdefault(r['team'], []).append(r['n'])
    man = man.set_index('team')
    for t, names in outs.items():
        have = [x for x in man.at[t, 'out'].split(';') if x] if t in man.index else []
        merged = ';'.join(have + [n for n in names if n not in have])
        if t in man.index: man.at[t, 'out'] = merged
        else: man.loc[t] = {'out': merged, 'add': '', 'qb': '', 'until': ''}
    man.reset_index()[['team', 'out', 'add', 'qb', 'until']].to_csv(NFL_CSV, index=False)
    print('NFL overrides ->', NFL_CSV, f'{sum(map(len, outs.values()))} ESPN outs on {len(outs)} teams + manual rows')

def main():
    try: prev = json.load(open(OUT_JSON))
    except Exception: prev = {}
    doc = {'made': NOW.strftime('%Y-%m-%d %H:%M UTC'), 'src': 'ESPN injuries feed + models/injuries/manual.json', 'feeds': {}, 'changes': []}
    for lg in LEAGUES:
        try:
            rows = judge(lg, scan_league(lg)); doc['feeds'][lg] = {'ok': True, 'fetched': doc['made'], 'entries': len(rows)}
        except Exception as e:
            old = prev.get(lg); ft = (prev.get('feeds', {}).get(lg) or {}).get('fetched')
            age = (NOW - datetime.datetime.strptime(ft, '%Y-%m-%d %H:%M UTC').replace(tzinfo=datetime.timezone.utc)).total_seconds() / 3600 if ft else 1e9
            if old and age <= MAX_AGE_H:
                rows = [r for t in old.values() for r in t]; doc['feeds'][lg] = {'ok': False, 'fetched': ft, 'fallback': f'last good scan, {age:.0f} h old', 'error': str(e)[:200]}
            else:
                rows = []; doc['feeds'][lg] = {'ok': False, 'fetched': ft, 'fallback': 'none (no injury input this run)', 'error': str(e)[:200]}
            print(f'::warning::{lg} injury feed failed ({e}); fallback:', doc['feeds'][lg]['fallback'])
        rows = manual(lg, [r for r in rows if r.get('src') != 'manual'])
        # keep only entries that change something, plus day-to-day ones we could not read (for the log)
        keep = [r for r in rows if r['p'] is not None or r['st'] == 'Day-To-Day']
        doc[lg] = {}
        for r in sorted(keep, key=lambda r: (r['team'], r['n'] or '')): doc[lg].setdefault(r['team'], []).append(r)
        # change log versus the previous scan
        was = {(t, r['n']): r.get('p') for t, v in (prev.get(lg) or {}).items() for r in v}
        now = {(r['team'], r['n']): r.get('p') for r in keep}
        for k in sorted(set(was) | set(now), key=str):
            a, b = was.get(k, 'absent'), now.get(k, 'absent')
            if a != b: doc['changes'].append({'lg': lg, 'team': k[0], 'n': k[1], 'from': a, 'to': b})
        (write_nba if lg == 'NBA' else write_nfl)(keep)
    with open(LOG, 'a') as f:
        for c in doc['changes']: f.write(json.dumps({'t': doc['made'], **c}) + '\n')
    json.dump(doc, open(OUT_JSON, 'w'), separators=(',', ':'))
    print(f"{len(doc['changes'])} status changes since the last scan ->", OUT_JSON)
    for c in doc['changes'][:40]: print('  ', c)

if __name__ == '__main__':
    try: main()
    except Exception as e:   # never block the chain: the exporters run without injury input as before
        print('::warning::injury scan failed:', e); sys.exit(0)
