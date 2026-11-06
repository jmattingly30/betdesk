"""Game slates for Bet Desk (J 2026-10-17: CFB and CBB fill their slate like NFL and NBA) -> app/data/game_slates.json.

python3 export_slates.py [nfl_games.csv]     # default /tmp/claude-0/nfldl/games.csv (models/nfl-player fetch_data.py)

Per sport, the upcoming games: id, kickoff (UTC), day (US Eastern), home, away, neutral. NFL games also carry the market
line nflverse publishes in games.csv (consensus spread, total, moneylines and their odds), which the app uses to flag
games where the model beats the line by 5%+ EV.
Live lines (J 2026-10-17): with ODDS_API_KEY set (an environment secret, never in code), every sport pulls The Odds API
once (h2h, spreads, totals, US books; 3 credits per sport, ~360 a month for 4 sports on the 500-credit free plan) and
saves the raw reply to data/odds/{SPORT}.json. Without the key, a saved reply under 30 hours old is used, so a separate
session that has the key can pull the odds first. Each game's ln is then the median across books (spread and total
points, and the odds at that point), and ln.bk keeps every book's line for single-book shopping later. NFL falls back
to nflverse when there is no odds reply.
Sources: nflverse games.csv; cfbfastR-data schedules (CFB, FBS games incl. FCS opponents); sportsdataverse ESPN
schedules (men's college basketball, NBA)."""
import sys, io, os, re, json, statistics, unicodedata, datetime as dt, urllib.request
from zoneinfo import ZoneInfo
import pandas as pd

HERE = __file__.rsplit('/', 1)[0]
OUT = f'{HERE}/data/game_slates.json'
ET = ZoneInfo('America/New_York')
NOW = dt.datetime.now(dt.timezone.utc)
TODAY = NOW.astimezone(ET).date().isoformat()
SV = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download'
# J 2026-11-06: the desk lists the whole day's games after they start (log a missed bet, review the data). Games that
# started in the last KEEP_H hours stay on the slate: finished ones from the schedule with their final score (fs = [home, away]),
# and every started game keeps its last pre-game line from the previous file (models/model-b/pregame.py).
sys.path.insert(0, f"{HERE}/../models/model-b")
from pregame import keep as keep_pregame, KEEP_H
RECENT = NOW - dt.timedelta(hours=KEEP_H)
SEASON_CFB, SEASON_CBB, SEASON_NBA = 2026, 2027, 2027
# J 2026-10-27 (line shopping): up to 10 named books cost the same credits as one region, so Pinnacle and BetOnline (the sharp
# prices that set fair odds in line_tools.py) come free. The other 8 are the books J can bet in Colorado (free feed only).
BOOKS = ['pinnacle', 'betonlineag', 'draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'espnbet', 'fanatics', 'betrivers', 'ballybet']
ODDS_KEYS = dict(NFL='americanfootball_nfl', CFB='americanfootball_ncaaf', CBB='basketball_ncaab', NBA='basketball_nba')
NFL_MASCOT = {**dict(Cardinals='ARI', Falcons='ATL', Ravens='BAL', Bills='BUF', Panthers='CAR', Bears='CHI', Bengals='CIN', Browns='CLE',
                  Cowboys='DAL', Broncos='DEN', Lions='DET', Packers='GB', Texans='HOU', Colts='IND', Jaguars='JAX', Chiefs='KC',
                  Raiders='LV', Chargers='LAC', Rams='LA', Dolphins='MIA', Vikings='MIN', Patriots='NE', Saints='NO', Giants='NYG',
                  Jets='NYJ', Eagles='PHI', Steelers='PIT', Seahawks='SEA', Buccaneers='TB', Titans='TEN', Commanders='WAS'), '49ers': 'SF'}
# Odds API school names the schedules spell differently (normalised, without the mascot)
ALIAS = {'southern mississippi': 'Southern Miss', 'umass': 'Massachusetts', 'citadel': 'The Citadel', 'sam houston state': 'Sam Houston',
         'appalachian state': 'App State', 'louisiana monroe': 'UL Monroe', 'ul monroe': 'UL Monroe', 'connecticut': 'UConn',
         'florida intl': 'Florida International', 'fiu': 'Florida International', 'texas el paso': 'UTEP', 'texas san antonio': 'UTSA'}

def norm(x):
    x = unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'\s+', ' ', x.replace('&', ' and ').replace("'", '').replace('.', '')).strip()

def get(url):
    with urllib.request.urlopen(url, timeout=90) as r: return r.read()

def et_day(ts): return ts.astimezone(ET).date().isoformat()

def nfl(path):
    import os
    g = pd.read_csv(path) if os.path.exists(path) else pd.read_csv(io.BytesIO(get('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv')))
    g = g[g.season == g.season.max()].copy()
    g['k'] = [pd.Timestamp(f'{d} {t if t == t else "13:00"}', tz=ET).tz_convert('UTC') for d, t in zip(g.gameday, g.gametime)]
    # done = final score posted, or kicked off 5+ hours ago (nflverse can lag hours on a Monday night score)
    fin = g.home_score.notna() & (g.k >= RECENT) & (g.k <= NOW)   # finished in the last KEEP_H hours: kept with the score
    g = g[(g.home_score.isna() & (g.k > NOW - pd.Timedelta(hours=5))) | fin]
    if not len(g): return None
    # J 2026-10-20 (be as early as possible on lines): this week's remaining games plus all of next week, so next week's
    # games and market lines are on the slate from Sunday night on, not only after Monday night's game
    fin = g.home_score.notna(); wk = int(g[~fin].week.min()) if (~fin).any() else int(g.week.max()); g = g[g.week.isin([wk, wk + 1]) | fin]
    out = []
    for r in g.itertuples():
        k = r.k
        ln = None
        if r.spread_line == r.spread_line:   # nflverse spread_line is home margin (home favored > 0); the app's home line is its negative
            f = lambda x: None if x != x else int(x)
            ln = dict(L=-float(r.spread_line), T=None if r.total_line != r.total_line else float(r.total_line),
                      oH=f(r.home_spread_odds), oA=f(r.away_spread_odds), oO=f(r.over_odds), oU=f(r.under_odds),
                      mlH=f(r.home_moneyline), mlA=f(r.away_moneyline))
        out.append(dict(id=r.game_id, k=k.isoformat().replace('+00:00', 'Z'), d=et_day(k), h=r.home_team, a=r.away_team,
                        n=int(getattr(r, 'location', 'Home') == 'Neutral'), ln=ln, w=int(r.week),
                        **({'fs': [int(r.home_score), int(r.away_score)]} if r.home_score == r.home_score else {})))
    return dict(week=wk, src='nflverse games.csv (consensus line)', games=out)

def cfb():
    s = pd.read_csv(io.BytesIO(get(f'https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_{SEASON_CFB}.csv')))
    s['k'] = pd.to_datetime(s.start_date, utc=True); s['d'] = s.k.map(et_day)
    up = s[(s.d >= TODAY) | ((s.k >= RECENT) & (s.k <= NOW))]
    if not len(up): return None
    wk = int(up[up.d >= TODAY].week.min()) if (up.d >= TODAY).any() else int(up.week.max()); up = up[(up.week == wk) | (up.d < TODAY)].sort_values('k')
    names = {t: [t] for t in set(s.home_team) | set(s.away_team)}
    fs = lambda r: {'fs': [int(r.home_points), int(r.away_points)]} if str(getattr(r, 'completed', '')).lower() == 'true' and r.home_points == r.home_points else {}
    return dict(week=wk, names=names, games=[dict(id=str(r.game_id), k=r.k.isoformat().replace('+00:00', 'Z'), d=r.d, h=r.home_team, a=r.away_team,
                                     n=int(bool(r.neutral_site)), **fs(r)) for r in up.itertuples()])

def odds_events(sp):
    """The Odds API reply for one sport: pulled now when the key is set, else a saved reply under 30 hours old.
    J 2026-10-27 (free feeds only): a reply op_live.py wrote from OddsPapi (free) under 6 hours old is used first, so
    The Odds API credits are only spent when the OddsPapi sweep failed or is stale (ODDS_FORCE=1 pulls anyway; ODDS_FORCE=CFB,CBB
    pulls only those)."""
    f = f'{HERE}/data/odds/{sp}.json'; key = os.environ.get('ODDS_API_KEY')
    force = os.environ.get('ODDS_FORCE', '')   # '1' = every sport; or a list like 'CFB,CBB' (noon pull, J 2026-10-28)
    if os.path.exists(f) and not (force == '1' or sp in force.split(',')):
        d = json.load(open(f))
        if d.get('src') == 'oddspapi' and NOW - dt.datetime.fromisoformat(d['made']) < dt.timedelta(hours=6):
            print(sp, 'odds from OddsPapi sweep', d['made'][:16]); return d['events'], d.get('left')
    # CBB opens 2026-11-02: no credits spent on its few early lines before then (OddsPapi covers it once books post)
    if key and not (sp == 'CBB' and NOW.date() < dt.date(2026, 11, 2)):
        url = (f'https://api.the-odds-api.com/v4/sports/{ODDS_KEYS[sp]}/odds?bookmakers={",".join(BOOKS)}&markets=h2h,spreads,totals'
               f'&oddsFormat=american&apiKey={key}')
        with urllib.request.urlopen(url, timeout=60) as r:
            ev = json.loads(r.read()); left, used = r.headers.get('x-requests-remaining'), r.headers.get('x-requests-used')
        os.makedirs(os.path.dirname(f), exist_ok=True)
        json.dump(dict(made=NOW.isoformat(), left=left, used=used, events=ev), open(f, 'w'), separators=(',', ':'))
        print(sp, 'odds pulled:', len(ev), 'games, credits left', left)
        return ev, left
    if os.path.exists(f):
        d = json.load(open(f))
        if NOW - dt.datetime.fromisoformat(d['made']) < dt.timedelta(hours=30): return d['events'], d.get('left')
    return None, None

def american(dec): return None if dec is None else (round((dec - 1) * 100) if dec >= 2 else round(-100 / (dec - 1)))
def decimal(a): return 1 + (a / 100 if a > 0 else 100 / -a)
def med_price(xs): return american(statistics.median(decimal(x) for x in xs)) if xs else None
def half(x): return round(x * 2) / 2

def consensus(e, h_name):
    """Median line across books from the home side's view (h_name = the Odds API name of the slate's home team)."""
    bk = {}; po = {}; sp, to, ml = [], [], []
    for b in e.get('bookmakers', []):
        M = {m['key']: m['outcomes'] for m in b.get('markets', [])}; row = [None] * 8
        # when this book first posted each line (OddsPapi only): [spread, total, moneyline], for the desk's Look ahead
        P = {m['key']: m.get('posted') for m in b.get('markets', [])}
        if any(P.values()): po[b['key']] = [P.get('spreads'), P.get('totals'), P.get('h2h')]
        for o in M.get('spreads', []):
            if o['name'] == h_name: row[0], row[1] = o.get('point'), o.get('price')
            else: row[2] = o.get('price')
        for o in M.get('totals', []):
            if o['name'] == 'Over': row[3], row[4] = o.get('point'), o.get('price')
            elif o['name'] == 'Under': row[5] = o.get('price')
        for o in M.get('h2h', []):
            if o['name'] == h_name: row[6] = o.get('price')
            elif o['name'] != 'Draw': row[7] = o.get('price')
        if any(v is not None for v in row): bk[b['key']] = row
    R = list(bk.values())
    pts = [r[0] for r in R if r[0] is not None and r[1] and r[2]]; tps = [r[3] for r in R if r[3] is not None and r[4] and r[5]]
    L = half(statistics.median(pts)) if pts else None; T = half(statistics.median(tps)) if tps else None
    at_L = [r for r in R if r[0] == L and r[1] and r[2]]; at_T = [r for r in R if r[3] == T and r[4] and r[5]]
    ln = dict(L=L, T=T, oH=med_price([r[1] for r in at_L]) or (-110 if L is not None else None),
              oA=med_price([r[2] for r in at_L]) or (-110 if L is not None else None),
              oO=med_price([r[4] for r in at_T]) or (-110 if T is not None else None),
              oU=med_price([r[5] for r in at_T]) or (-110 if T is not None else None),
              mlH=med_price([r[6] for r in R if r[6] and r[7]]), mlA=med_price([r[7] for r in R if r[6] and r[7]]),
              nb=len(bk), bk=bk, **({'po': po} if po else {}))
    return ln if (L is not None or T is not None or ln['mlH']) else None

def add_lines(sp, x, names):
    """Attach Odds API consensus lines to the slate games. names: slate team -> list of full names to match on."""
    if not x or not x['games']: return 0   # no slate, no credits spent
    ev, left = odds_events(sp)
    if ev is None: return 0
    by_norm = sorted(((('~' if n[0] == '~' else '') + norm(n.lstrip('~')), t) for t, ns in names.items() for n in ns), key=lambda z: -len(z[0]))
    def team(o):
        if sp == 'NFL': return NFL_MASCOT.get(o.split()[-1])
        n = norm(o)
        for a, t in ALIAS.items():
            if n == a or n.startswith(a + ' '): return t
        for k, t in by_norm:
            if n == k or n.startswith(k + ' ') or (k[0] == '~' and n.endswith(' ' + k[1:])): return t
    got = 0
    for e in ev:
        k = pd.Timestamp(e['commence_time'])
        if k <= NOW: continue   # games under way carry live in-game odds, not a pregame line
        h, a = team(e['home_team']), team(e['away_team'])
        near = [g for g in x['games'] if abs((pd.Timestamp(g['k']) - k).total_seconds()) < 12 * 3600]
        g = next((g for g in near if {g['h'], g['a']} == {h, a}), None) or next(
            (g for g in near if abs((pd.Timestamp(g['k']) - k).total_seconds()) < 3 * 3600 and (h in (g['h'], g['a']) or a in (g['h'], g['a']))), None)
        if not g: continue
        flip = g['h'] == a or (g['h'] != h and g['a'] == h)
        ln = consensus(e, e['away_team'] if flip else e['home_team'])
        if ln: g['ln'] = ln; got += 1
    x['src'] = f"{'OddsPapi' if ev and ev[0].get('src') == 'oddspapi' else 'The Odds API'} ({len(ev)} games pulled, median of books)"; x['credits_left'] = left
    print(sp, 'lines on', got, 'of', len(x['games']), 'slate games')
    return got

def espn(url, days, name_col, mascot=False):
    s = pd.read_parquet(io.BytesIO(get(url)))
    s['k'] = pd.to_datetime(s.date, utc=True); s['d'] = s.k.map(et_day)
    end = (NOW.astimezone(ET).date() + dt.timedelta(days=days)).isoformat()
    done = s.status_type_completed.astype(bool)
    up = s[((s.d >= TODAY) & (s.d <= end) & ~done) | (done & (s.k >= RECENT) & (s.k <= NOW))].sort_values('k')
    if not len(up): return None
    names = {}
    for r in s.to_dict('records'):
        for side in ('home_', 'away_'):
            names.setdefault(r[side + name_col], {r[side + 'display_name'], r[side + 'location'] + ' ' + r[side + 'name']} |
                             ({'~' + r[side + 'name']} if mascot else set()))   # ~ = match on the nickname (NBA: "LA" vs "Los Angeles")
    fs = lambda r: {'fs': [int(r['home_score']), int(r['away_score'])]} if r['status_type_completed'] and r.get('home_score') == r.get('home_score') and r.get('home_score') is not None else {}
    return dict(games=[dict(id=str(r['game_id']), k=r['k'].isoformat().replace('+00:00', 'Z'), d=r['d'], h=r['home_' + name_col],
                            a=r['away_' + name_col], n=int(bool(r['neutral_site'])), **fs(r)) for r in up.to_dict('records')], names=names)

if __name__ == '__main__':
    J = dict(made=NOW.strftime('%Y-%m-%d %H:%M UTC'))
    jobs = dict(NFL=lambda: nfl(sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-0/nfldl/games.csv'), CFB=cfb,
                CBB=lambda: espn(f'{SV}/espn_mens_college_basketball_schedules/mbb_schedule_{SEASON_CBB}.parquet', 6, 'location'),
                NBA=lambda: espn(f'{SV}/espn_nba_schedules/nba_schedule_{SEASON_NBA}.parquet', 6, 'abbreviation', mascot=True))
    for sp, f in jobs.items():
        try: x = f()
        except Exception as e: print(sp, 'failed:', e); x = None
        try:   # NFL games the odds reply misses keep their nflverse line
            if x: add_lines(sp, x, x.get('names', {}))
        except Exception as e: print(sp, 'odds failed:', e)
        if x:   # started games keep their last pre-game line (the odds feed drops it at kickoff) and come back if the feed dropped them
            x.pop('names', None)
            x['games'] = keep_pregame(OUT, x['games'], kick=lambda i, r: r.get('k'), field=f'{sp}.games',
                                      merge=lambda o, n: {**n, **({'ln': o['ln']} if o.get('ln') and not n.get('ln') else {})})
            J[sp] = x
        print(sp, len(x['games']) if x else 0, 'games', f"week {x['week']}" if x and 'week' in x else '')
    json.dump(J, open(OUT, 'w'), separators=(',', ':'))
    print('wrote', OUT)
