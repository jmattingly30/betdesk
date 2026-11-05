"""Grade every Bet Desk prop projection against the box score, logged as a bet or not (Bet Desk audit M6, 2026-10-17).

python3 grade.py            # snapshot today's props files, then grade everything that has a result

Step 1 (snapshot): each player-stat in app/data/{nfl,nba,nhl}_props.json is saved to history/{SPORT}.json with its
projection, our fair line (the half-point line closest to 50%), the over chance there, and the Bet Desk flag (15% rule,
same as flagOf in bet-desk.html). A later refresh replaces a game's projections only until that game has a box score.
Step 2 (grade): box scores (NFL nflverse weekly stats, NHL and NBA sportsdataverse player boxes) are matched on game id +
team + player name, and app/data/scorecard.json gets per sport and stat: count, average miss, bias, how often the result
went over our fair line vs how often we said it would, a calibration table, and how flagged players did against their
average. The Bet Desk Results tab reads that file.
Step 3 (games, 2026-10-17): snap_games.js saves every upcoming game's projected score from the app itself to
history/GAMES.json.gz; here each one with a final score (NFL nflverse games.csv, NBA/CBB ESPN schedules, CFB ESPN
schedule) is scored: margin, total and team-score misses, winner picked, the market line's misses on the same games,
and the record of games flagged at 5%+ EV. app/data/player_results.json keeps the last 7 days of graded props for the
Results tab's player lookup.
Step 4 (tracking, 2026-10-18, J: chart success by team and by player): scorecard.json gets `teams` (each team's points
scored and allowed vs our projected score, game by game, from the game snapshots), `team_props` (each team's players,
and the players facing each defence, vs our prop projections, in %), `trend` (daily accuracy per sport) and a `slope`
per stat/sport (actual on projected: 1 = right-sized, under 1 = we over-react to differences). player_results.json gets
`track`: each player-stat's actual minus projected over games that meet the playtime minimum (PLAY_MIN), also on
comparable-workload games only (75-133% of the expected minutes/TOI/snaps, J's workload rule) and re-scaled to the
playtime he actually got, so a miss can be told apart as role (minutes) or rate.
Step 5 (matchup, J 2026-10-18 option "buckets + shadow"): scorecard `matchup` = our prop miss by opponent tier (thirds of
what that defence allowed the position group in its earlier games, 3+ needed); `gmatchup` = team score miss by the
opponent's defence / offence / pace tier from earlier games; `shadow` = NFL scheme shadow terms (models/nfl-player/
shadow.py, run before this) vs the official projection on the same games. Flat tiers = our matchup handling works.
Step 6 (line flags, Bet Desk v71, 2026-10-28): scorecard `lineflags` grades snap_flags.js rows (history/FLAGS.json.gz) at
the flagged line and price (W/L/P, units, ROI by sport x kind flag/proven/check, daily units) and keeps the last pregame
price per flag in history/FLAGS_CLOSE.json.gz for CLV. The old average-based `flags` grading is unchanged."""
import json, os, sys, io, gzip, datetime as dt, urllib.request
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.normpath(f'{HERE}/../../app/data')
HIST = f'{HERE}/history'
os.makedirs(HIST, exist_ok=True)
SEASON_NFL, SEASON_NHL, SEASON_NBA = 2026, 2027, 2027
SRC = {'NFL': f'https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{SEASON_NFL}.parquet',
       'NHL': f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nhl_player_boxscores/player_box_{SEASON_NHL}.parquet',
       'NBA': f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_{SEASON_NBA}.parquet'}

# the Bet Desk flag rule (flagOf/flagRole in bet-desk.html, FLAG_MIN/FLAG_FLOOR), at the default 15%
FLAG_MIN = {'NFL': dict(rec_yds=8, rush_yds=8, rush_rec_yds=10, pass_yds=20, targets=1, rec=0.8, carries=2, pass_att=3, cmp=2, pass_td=0.25, td=0.1),
            'NBA': dict(pts=2.5, reb=1, ast=1, fg3m=0.5, pra=3, stl=0.3, blk=0.3, tov=0.4), 'NHL': dict(sog=0.6, points=0.2, goals=0.12, assists=0.15, blk=0.5, hits=0.6, saves=2)}
FLAG_FLOOR = {'NFL': dict(rec_yds=20, rush_yds=20, rush_rec_yds=25, pass_yds=150, targets=3, rec=2, carries=6, pass_att=20, cmp=12, pass_td=0.8, td=0.25),
              'NBA': dict(pts=8, reb=3, ast=2, fg3m=1, pra=12, stl=0.6, blk=0.5, tov=1), 'NHL': dict(sog=1.5, points=0.4, goals=0.2, assists=0.25, blk=1, hits=1.5, saves=20)}

def flag(sp, stat, s, thr=0.15):
    a = s.get('a')
    if not a or not (a[2] >= 3): return None
    if sp == 'NFL' and stat == 'td' and not (s['m'] > 0.5): return None
    mn, fl = FLAG_MIN[sp].get(stat, 1), FLAG_FLOOR[sp].get(stat, 0)
    hi, lo = max(a[0], a[1]), min(a[0], a[1])
    if hi > 0 and s['m'] >= (1 + thr) * hi and s['m'] - hi >= mn and s['m'] >= fl: return ('over', hi)
    if lo > 0 and s['m'] <= (1 - thr) * lo and lo - s['m'] >= mn and lo >= fl: return ('under', lo)
    return None

def role_ok(sp, p):
    if sp == 'NBA':
        try: return float(p['p'].split('·')[-1].split()[0]) >= 15
        except Exception: return True
    if sp == 'NHL': return p.get('p') == 'G' or not (p.get('toi', 99) < 12)
    return True

def fair(c):
    """half-point ladder line whose over chance is closest to 50%, and that chance"""
    L, pv = min(((x, y) for x, y in c if float(x) % 1 == 0.5) or c, key=lambda t: abs(t[1] - 0.5))
    return float(L), float(pv)

def fetch(sp, url=None):
    try:
        with urllib.request.urlopen(url or SRC[sp], timeout=60) as r: return pd.read_parquet(io.BytesIO(r.read()))
    except Exception as e:
        print(sp, 'box scores not available:', e); return None

SNAPS_NFL = f'https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{SEASON_NFL}.parquet'
# playtime minimum for player tracking (actual workload: NBA minutes, NHL TOI minutes, NFL offensive snap share)
PLAY_MIN = {'NBA': 15, 'NHL': 12, 'NHL_G': 40, 'NFL': 0.25, 'NFL_QB': 0.5}

def toi_min(s):
    try: m, x = str(s).split(':'); return int(m) + int(x) / 60
    except Exception: return None

def results(sp, box):
    """{(game_id, team, name): {stat: value}} in the props files' stat keys"""
    out = {}
    if box is None or not len(box): return out
    if sp == 'NFL':
        b = box[box.season_type == 'REG'] if 'season_type' in box else box
        sn = fetch('NFL', SNAPS_NFL); snap = {}
        if sn is not None:
            for r in sn.itertuples(): snap[(str(r.game_id), r.team, str(r.player))] = r.offense_pct
        for r in b.itertuples():
            v = dict(pass_att=r.attempts, cmp=r.completions, pass_yds=r.passing_yards, pass_td=r.passing_tds, carries=r.carries, rush_yds=r.rushing_yards,
                     rec=r.receptions, targets=r.targets, rec_yds=r.receiving_yards, rush_rec_yds=(r.rushing_yards or 0) + (r.receiving_yards or 0),
                     td=(r.rushing_tds or 0) + (r.receiving_tds or 0))
            w = snap.get((str(r.game_id), r.team, str(r.player_display_name)))
            if w is not None: v['_w'] = w
            for nm in {r.player_display_name, r.player_name}: out[(str(r.game_id), r.team, str(nm))] = v
    elif sp == 'NHL':
        for r in box.itertuples():
            g = r.position not in ('C', 'L', 'R', 'D')       # the box score leaves goalies' position blank (fixed 2026-10-18: saves were never graded)
            if g and not r.starter: continue
            v = dict(saves=r.saves) if g else dict(sog=r.shots_on_goal, points=r.points, goals=r.goals, assists=r.assists, blk=r.blocked_shots, hits=r.hits)
            v['_w'] = toi_min(r.toi)
            out[(str(r.game_id), r.team_abbrev, str(r.player_name) + (' (G)' if g else ''))] = v
    elif sp == 'NBA':
        cols = set(box.columns)
        for r in box.to_dict('records'):
            if not r.get('minutes'): continue
            g = lambda *k: next((r[x] for x in k if x in cols and r.get(x) is not None), None)
            pts, reb, ast = g('points'), g('rebounds'), g('assists')
            v = dict(pts=pts, reb=reb, ast=ast, fg3m=g('three_point_field_goals_made'), stl=g('steals'), blk=g('blocks'), tov=g('turnovers'),
                     pra=(pts or 0) + (reb or 0) + (ast or 0) if pts is not None else None, _w=g('minutes'))
            # keyed by DATE, not ESPN game id: nba_props.json numbers its games 1, 2, ... per export (fixed 2026-10-21), and a team plays once a day
            out[(str(g('game_date'))[:10], str(g('team_abbreviation')), str(g('athlete_display_name')))] = v
    def num(x):
        try: x = float(x); return x if x == x else None
        except Exception: return None
    return {k: {s: num(x) for s, x in v.items() if num(x) is not None} for k, v in out.items()}

def exp_work(sp, p):
    """the props file's expected workload for this player: NBA minutes, NHL TOI, NFL snap share"""
    wl = p.get('wl') or {}
    if wl.get('exp') is not None: return wl['exp']
    if sp == 'NHL': return p.get('toi')
    if sp == 'NBA':
        try: return float(p['p'].split('·')[-1].split()[0])
        except Exception: return None
    return None

def snapshot(sp, res):
    f = f'{APP}/{sp.lower()}_props.json'
    if not os.path.exists(f): return
    d = json.load(open(f)); H = load(sp); played = {k[0] for k in res}; n = 0
    for p in d['players']:
        made_day = str(d.get('made', ''))[:10]
        if gkey(sp, p) in played:                                # a game with a result keeps the projection it had before it
            if not p.get('d') or not made_day or made_day > p['d'] or any(k.startswith(f"{gkey(sp, p)}|{p['t']}|{p['n']}|") for k in H): continue
        for st, s in p['s'].items():
            L, po = fair(s['c']); fg = flag(sp, st, s) if role_ok(sp, p) else None
            H[f"{gkey(sp, p)}|{p['t']}|{p['n']}|{st}"] = dict(sp=sp, g=gkey(sp, p), t=p['t'], o=p.get('o'), n=p['n'], d=p.get('d') or f"week {d.get('week')}", stat=st,
                m=s['m'], L=L, po=round(po, 4), cs=s.get('cs', 0), flag=fg[0] if fg else None, base=fg[1] if fg else None,
                avg=(s.get('a') or [None, None])[:2], made=d.get('made'), we=exp_work(sp, p), pos=str(p.get('p') or '').split('·')[0].strip(), h=p.get('h'))
            n += 1
    json.dump(H, gzip.open(f'{HIST}/{sp}.json.gz', 'wt')); print(sp, 'snapshot', n, 'projections, history', len(H))

def gkey(sp, p):
    """the game part of a history key: NBA props carry export-local game numbers, so NBA uses the game date"""
    return str(p.get('d')) if sp == 'NBA' and p.get('d') else str(p['g'])

def load(sp):
    f = f'{HIST}/{sp}.json.gz'
    H = json.load(gzip.open(f, 'rt')) if os.path.exists(f) else {}
    if sp == 'NBA':      # 2026-10-21: re-key older rows saved under the export-local game number (latest snapshot wins)
        M = {}
        for k, h in sorted(H.items(), key=lambda kv: str(kv[1].get('made'))):
            if h.get('d'): h['g'] = str(h['d'])
            M[f"{h['g']}|{h['t']}|{h['n']}|{h['stat']}"] = h
        H = M
    return H

GAME_SRC = {'NFL': 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv',
            'NBA': f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_schedules/nba_schedule_{SEASON_NBA}.parquet',
            'CFB': f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_schedules/cfb_schedule_2026.parquet',
            'CBB': f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/mbb_schedule_2027.parquet'}

MLB_SLATE, MLB_FIN = f'{APP}/mlb_slate.json', f'{HIST}/MLB_FINALS.json.gz'

# J 2026-10-19: v2 retired (it misread his efficiency system); then J's rebuilt efficiency model 'eff' became primary.
# Primary = the first of MLB_PRIMARY present (eff, else v3 for older snapshots); every other model except v2 is graded beside it.
MLB_PRIMARY = ['eff', 'v3']

def mlb_models(m):
    """(primary [home, away] runs, {other model: [home, away]}, primary key) from an Apps Script models dict; v2 ignored"""
    m = {k: v for k, v in (m or {}).items() if k != 'v2' and isinstance(v, dict) and v.get('runsHome') is not None}
    pk = next((k for k in MLB_PRIMARY if k in m), None)
    if not pk: return None, {}, None
    p = m[pk]
    return [p['runsHome'], p['runsAway']], {k: [v['runsHome'], v['runsAway']] for k, v in m.items() if k != pk}, pk

def snapshot_cbb_shadow():
    """CBB (J 2026-10-27): the v2 rebuild is Bet Desk's main CBB model (snapshot hs/as). The old CBB model rides along as a
    shadow, mv['old'] = app/data/cbb_v2.json games[id].today_hs/today_as, written only before tip-off so it stays pregame."""
    f, src = f'{HIST}/GAMES.json.gz', f'{APP}/cbb_v2.json'
    if not (os.path.exists(f) and os.path.exists(src)): return
    H = json.load(gzip.open(f, 'rt')); G = (json.load(open(src)).get('games') or {}); now = dt.datetime.now(dt.timezone.utc).isoformat(); n = 0
    for h in H.values():
        g = G.get(str(h.get('id')))
        if h['sp'] != 'CBB' or not g or g.get('today_hs') is None or (h.get('k') and h['k'] <= now): continue
        h.setdefault('mv', {})['old'] = [g['today_hs'], g['today_as']]; n += 1
    json.dump(H, gzip.open(f, 'wt')); print('CBB old-model shadow on', n, 'upcoming games')

def snapshot_mlb():
    """MLB (2026-10-18, J): the cloud cannot reach MLB Stats API, so J's Apps Script is the source for both sides. The
    refresh copies mlb_slate.json from J's Drive folder "Bet Desk MLB" to app/data/; each game not yet started is saved
    with the primary model (MLB_PRIMARY: eff, else v3; v2 retired 2026-10-19) as hs/as and the other models in mv; the slate's `finals` (last 3 days, added to
    exportSlate in Server.gs 2026-10-18) are kept in history/MLB_FINALS.json.gz. From 2026-10-19 each final carries `pregame`
    (last projection before first pitch), which replaces the morning snapshot for grading; null pregame keeps the snapshot."""
    if not os.path.exists(MLB_SLATE): print('MLB: no mlb_slate.json yet (Apps Script feed not live)'); return
    d = json.load(open(MLB_SLATE)); f = f'{HIST}/GAMES.json.gz'
    H = json.load(gzip.open(f, 'rt')) if os.path.exists(f) else {}
    now = dt.datetime.now(dt.timezone.utc).isoformat(); n = 0
    for g in d.get('games', []):
        pr, mv, pk = mlb_models(g.get('models'))
        if g.get('status') != 'ok' or pr is None or not g.get('startUtc') or g['startUtc'] <= now: continue
        H[f"MLB|{g['gamePk']}"] = dict(sp='MLB', id=str(g['gamePk']), d=g.get('date') or d.get('date'), k=g['startUtc'], h=g['home'], a=g['away'], n=1,
                                        hs=pr[0], **{'as': pr[1]}, mode='model', ln={}, e=None, made=d.get('updatedAt'), mv=mv, primary=pk,
                                        sps=[g.get('homeSP'), g.get('awaySP')])
        n += 1
    F = json.load(gzip.open(MLB_FIN, 'rt')) if os.path.exists(MLB_FIN) else {}
    for x in d.get('finals') or []: F[str(x['gamePk'])] = x
    # 2026-10-19: finals[].pregame = the Apps Script's LAST projection before first pitch (Server.gs, MLB re-audit thread).
    # That is what we grade; the morning games[] snapshot above is only the fallback when pregame is null.
    npg = 0
    for k, x in F.items():
        pg = x.get('pregame') or {}; pr, mv, pk = mlb_models(pg.get('models'))
        if pr is None: continue
        H[f'MLB|{k}'] = dict(sp='MLB', id=str(k), d=x.get('date'), k=pg.get('startUtc') or x.get('date'), h=x['home'], a=x['away'], n=1,
                             hs=pr[0], **{'as': pr[1]}, mode='model', ln={}, e=None, made=pg.get('madeAt'), src='pregame', mv=mv, primary=pk,
                             sps=[pg.get('homeSP'), pg.get('awaySP')], lineups=pg.get('lineupsPosted'))
        npg += 1
    json.dump(H, gzip.open(f, 'wt'))
    json.dump(F, gzip.open(MLB_FIN, 'wt')); print('MLB snapshot', n, 'upcoming games;', npg, 'graded from pregame; finals kept', len(F))

def finals(sp):
    """{game id: (home score, away score)} for finished games"""
    if sp == 'MLB':
        F = json.load(gzip.open(MLB_FIN, 'rt')) if os.path.exists(MLB_FIN) else {}
        return {k: (float(x['homeRuns']), float(x['awayRuns'])) for k, x in F.items()}
    try:
        with urllib.request.urlopen(GAME_SRC[sp], timeout=90) as r: b = r.read()
    except Exception as e:
        print(sp, 'game results not available:', e); return {}
    if sp == 'NFL':
        g = pd.read_csv(io.BytesIO(b)); g = g[g.home_score.notna()]
        return {str(x.game_id): (float(x.home_score), float(x.away_score)) for x in g.itertuples()}
    g = pd.read_parquet(io.BytesIO(b))
    done = g.status.eq('STATUS_FINAL') if 'status' in g else g.status_type_completed.astype(bool)
    g = g[done]
    return {str(x.game_id): (float(x.home_score), float(x.away_score)) for x in g.itertuples()}

def grade_games():
    f = f'{HIST}/GAMES.json.gz'
    if not os.path.exists(f): return [], [], {}, {}, []
    H = json.load(gzip.open(f, 'rt')); out, recent, counts, pending, ALL = [], [], {}, {}, []
    for sp in ['NFL', 'NBA', 'CFB', 'CBB', 'MLB']:
        rows = [h for h in H.values() if h['sp'] == sp]
        if not rows: continue
        F = finals(sp); D = []
        for h in rows:
            y = F.get(str(h['id']))
            if not y: continue
            yh, ya = y; pm, pt, ym, yt = h['hs'] - h['as'], h['hs'] + h['as'], yh - ya, yh + ya
            ln = h.get('ln') or {}
            e, ew = h.get('e'), None
            if e:   # side A = home (spread, moneyline) or over (total); push = None
                if e['kind'] == 'spread': v = ym + float(e['line'])
                elif e['kind'] == 'total': v = yt - float(e['line'])
                else: v = ym
                if e['side'] == 'B': v = -v
                ew = None if v == 0 else int(v > 0)
            D.append(dict(h, yh=yh, ya=ya, em=abs(pm - ym), et=abs(pt - yt), es=(abs(h['hs'] - yh) + abs(h['as'] - ya)) / 2,
                          win=None if ym == 0 else int((pm > 0) == (ym > 0)),
                          lm=abs(-ln['L'] - ym) if ln.get('L') is not None else None, lt=abs(ln['T'] - yt) if ln.get('T') is not None else None,
                          lwin=None if ln.get('L') in (None, 0) or ym == 0 else int((-ln['L'] > 0) == (ym > 0)), ew=ew))
        pending[sp] = len(rows) - len(D); counts[sp] = len(D)
        if not D: continue
        ALL += D
        X = pd.DataFrame(D); L = X[X.lm.notna()]; E = X[X.e.notna() & X.ew.notna()] if 'e' in X else X.iloc[:0]
        pr = lambda o: (-1.0 if o is None else (100 / -o if o < 0 else o / 100))
        units = float(sum(pr(e['odds']) if w == 1 else -1 for e, w in zip(E.e, E.ew))) if len(E) else 0.0
        out.append(dict(sport=sp, model='Bet Desk', n=int(len(X)), mae_margin=round(float(X.em.mean()), 2), mae_total=round(float(X.et.mean()), 2),
                        mae_team=round(float(X.es.mean()), 2), slope=round(float(np.polyfit(X.hs - X['as'], X.yh - X.ya, 1)[0]), 3) if len(X) >= 20 else None, win_acc=round(float(X.win.dropna().mean()), 4) if X.win.notna().any() else None,
                        n_line=int(len(L)), mae_margin_line=round(float(L.lm.mean()), 2) if len(L) else None,
                        mae_total_line=round(float(L['lt'].dropna().mean()), 2) if L['lt'].notna().any() else None,
                        win_acc_line=round(float(L.lwin.dropna().mean()), 4) if len(L) and L.lwin.notna().any() else None,
                        edges=dict(n=int(len(E)), w=int((E.ew == 1).sum()), l=int((E.ew == 0).sum()), p=int(X.e.notna().sum() - len(E)) if 'e' in X else 0, units=round(units, 2))))
        if sp == 'MLB' and 'primary' in X: out[-1]['primary'] = {k: int(v) for k, v in X.primary.fillna('v3').value_counts().items()}   # which model the row graded
        if 'mv' in X:     # other models on the same games: MLB (primary eff, else v3; v2 retired), CBB 'old' shadow (primary = v2 rebuild, 2026-10-27)
            for v in sorted({k for m in X.mv if isinstance(m, dict) for k in m}):
                Z = [(m[v], yh, ya) for m, yh, ya in zip(X.mv, X.yh, X.ya) if isinstance(m, dict) and v in m]
                if not Z: continue
                ph, pa, yh, ya = (np.array([z[0][0] for z in Z]), np.array([z[0][1] for z in Z]), np.array([z[1] for z in Z]), np.array([z[2] for z in Z]))
                w = [(a > b) == (c > d) for a, b, c, d in zip(ph, pa, yh, ya) if c != d]
                out.append(dict(sport=sp, model=v, n=len(Z), mae_margin=round(float(np.abs((ph - pa) - (yh - ya)).mean()), 2), mae_total=round(float(np.abs((ph + pa) - (yh + ya)).mean()), 2),
                                mae_team=round(float((np.abs(ph - yh) + np.abs(pa - ya)).mean() / 2), 2), win_acc=round(float(np.mean(w)), 4) if w else None))
        for x in X.sort_values('k', ascending=False).head(40).to_dict('records'):
            recent.append(dict(sp=sp, d=x['d'], h=x['h'], a=x['a'], ph=x['hs'], pa=x['as'], yh=int(x['yh']), ya=int(x['ya']),
                               e=x['e']['name'] if isinstance(x.get('e'), dict) else None, ew=None if x['ew'] is None or x['ew'] != x['ew'] else int(x['ew'])))
    return out, recent, counts, pending, ALL

def grade(sp, res):
    H = load(sp); rows = []
    for k, h in H.items():
        r = res.get((h['g'], h['t'], h['n']), {}); v = r.get(h['stat'])
        if v is None: continue
        rows.append(dict(h, y=v, w=r.get('_w')))
    if not rows: return [], [], [], 0, None, [], None
    D = pd.DataFrame(rows); props, cal, fl = [], [], []
    for st, x in D.groupby('stat'):
        e = x.m - x.y
        sl = float(np.polyfit(x.m, x.y, 1)[0]) if len(x) >= 30 and x.m.std() > 0 else None
        base = x.avg.map(lambda a: a[1] if isinstance(a, list) and len(a) > 1 and a[1] is not None else (a[0] if isinstance(a, list) and a and a[0] is not None else None))
        ok = base.notna()
        props.append(dict(sport=sp, stat=st, n=int(len(x)), mae=round(float(e.abs().mean()), 3), bias=round(float(e.mean()), 3),
                          hit=round(float((x.y > x.L).mean()), 4), exp=round(float(x.po.mean()), 4),
                          mae_sub=round(float(e[ok].abs().mean()), 3) if ok.any() else None,
                          mae_avg=round(float((base[ok].astype(float) - x.y[ok]).abs().mean()), 3) if ok.any() else None,
                          slope=round(sl, 3) if sl is not None else None))
    D['bin'] = (D.po * 10).clip(0, 9.999).astype(int)
    for b, x in D.groupby('bin'):
        cal.append(dict(sport=sp, lo=b / 10, n=int(len(x)), said=round(float(x.po.mean()), 4), went_over=round(float((x.y > x.L).mean()), 4)))
    F = D[D.flag.notna()]
    for dr, x in F.groupby('flag'):
        hit_avg = (x.y > x.base) if dr == 'over' else (x.y < x.base)
        hit_fair = (x.y > x.L) if dr == 'over' else (x.y < x.L)
        fl.append(dict(sport=sp, dir=dr, n=int(len(x)), hit_avg=round(float(hit_avg.mean()), 4), hit_fair=round(float(hit_fair.mean()), 4)))
    days = sorted({d for d in D.d.astype(str) if d[:2] == '20'})[-7:]
    R = D[D.d.astype(str).isin(days)]
    pr = [[sp, x['d'], x['n'], x['t'] + (' vs ' + x['o'] if x.get('o') else ''), x['stat'], round(float(x['m']), 2), x['y']] for x in R.to_dict('records')]
    if sp == 'NFL': pr = [[sp, x['d'], x['n'], x['t'] + (' vs ' + x['o'] if x.get('o') else ''), x['stat'], round(float(x['m']), 2), x['y']] for x in D[D.d == D.d.max()].to_dict('records')]
    return props, cal, fl, len(D), str(D.d.max()), pr, D

def play_ok(sp, x):
    """meets the playtime minimum (unknown workload counts as yes; NFL snaps can lag the box score)"""
    w = x.w
    if sp == 'NBA': mn = PLAY_MIN['NBA']
    elif sp == 'NHL': mn = PLAY_MIN['NHL_G'] if x.stat == 'saves' else PLAY_MIN['NHL']
    else: mn = PLAY_MIN['NFL_QB'] if x.stat in ('pass_att', 'cmp', 'pass_yds', 'pass_td') else PLAY_MIN['NFL']
    return pd.isna(w) or w >= mn

def track(sp, D):
    """player-by-player and team-by-team over/under vs our projection, and the daily trend"""
    if D is None or not len(D): return [], [], []
    D = D.copy(); D['day'] = D.d.astype(str)
    D['play'] = [play_ok(sp, x) for x in D.itertuples()]
    # expected workload: the props file's, else the player's own median in graded games
    we = pd.to_numeric(D.we, errors='coerce') if 'we' in D else pd.Series(np.nan, index=D.index)
    D['we'] = we.fillna(D.groupby('n').w.transform('median'))
    ratio = D.w / D.we
    D['norm'] = ratio.between(0.75, 1.33)
    D['m_adj'] = np.where(ratio.notna() & (D.we > 0), D.m * ratio.clip(0, 2), np.nan)   # the projection at the playtime he got
    D['over'] = D.y - D.m
    P = D[D.play]; players = []
    for (n, t), x in P.groupby(['n', 't']):
        st = {}
        for s, z in x.groupby('stat'):
            z = z.sort_values('day'); zn = z[z.norm]
            r2 = lambda v: None if v != v else round(float(v), 2)
            # [games, avg projected, avg actual, actual-projected, comparable-workload games, their actual-projected,
            #  actual minus the projection re-scaled to the playtime he got, % over our fair line, our average over chance, last 10 [day, proj, actual]]
            st[s] = [int(len(z)), r2(z.m.mean()), r2(z.y.mean()), r2(z.over.mean()), int(len(zn)), r2(zn.over.mean()) if len(zn) else None,
                     r2((z.y - z.m_adj).mean()), round(float((z.y > z.L).mean()), 3), round(float(z.po.mean()), 3),
                     [[str(d)[5:] if str(d)[:2] == '20' else str(d).replace('week ', 'w'), r2(m), y] for d, m, y in zip(z.day, z.m, z.y)][-10:] if len(z) >= 2 else []]
        players.append(dict(sp=sp, n=n, t=t, pos=x.pos.iloc[-1] if 'pos' in x and isinstance(x.pos.iloc[-1], str) else None,
                            g=int(x.g.nunique()), we=round(float(x.we.mean()), 2) if x.we.notna().any() else None,
                            w=round(float(x.w.mean()), 2) if x.w.notna().any() else None, st=st))
    # teams: every projected player (team totals), as % over our projection, for the team's players and for the players it faced
    teams = []
    for side, key in (('for', 't'), ('ag', 'o')):
        for t, x in D[D[key].notna()].groupby(key):
            ng = x.g.nunique()      # per stat: [projected per game, actual per game, % over projection]
            st = {s: [round(float(z.m.sum() / ng), 2), round(float(z.y.sum() / ng), 2), round(float(z.y.sum() / z.m.sum() - 1), 3)] for s, z in x.groupby('stat') if z.m.sum() > 0}
            teams.append(dict(sp=sp, t=t, side=side, g=int(ng), st=st))
    # daily: unit-free so stats can be pooled; log loss and how often we beat the player's season average
    trend = []
    base = D.avg.map(lambda a: a[1] if isinstance(a, list) and len(a) > 1 and a[1] is not None else (a[0] if isinstance(a, list) and a else None))
    D['beat'] = np.where(base.notna(), (D.m - D.y).abs() < (base.astype(float) - D.y).abs(), np.nan)
    D['tie'] = np.where(base.notna(), (D.m - D.y).abs() == (base.astype(float) - D.y).abs(), np.nan)
    po = D.po.clip(0.01, 0.99); o = (D.y > D.L).astype(float)
    D['ll'] = -(o * np.log(po) + (1 - o) * np.log(1 - po))
    for day, x in D[D.day.str[:2] == '20'].groupby('day'):
        b = x[x.beat.notna() & (x.tie == 0)]
        trend.append(dict(sp=sp, d=day, n=int(len(x)), ll=round(float(x.ll.mean()), 4), beat_avg=round(float(b.beat.mean()), 4) if len(b) else None,
                          pct=round(float(x.y.sum() / x.m.sum() - 1), 4) if x.m.sum() > 0 else None))
    return players, teams, trend

LEAGUE = {}    # soccer team -> league name, filled by soccer_games()

def track_games(H_rows):
    """team by team: points scored and allowed vs our projected score; daily game accuracy"""
    X = pd.DataFrame(H_rows)
    if not len(X): return [], [], [], []
    teams = []
    for sp, z in X.groupby('sp'):
        long = pd.concat([pd.DataFrame(dict(t=z.h, o=z.a, d=z.d, pf=z.hs, af=z.yh, pa=z['as'], aa=z.ya, home=1)),
                          pd.DataFrame(dict(t=z.a, o=z.h, d=z.d, pf=z['as'], af=z.ya, pa=z.hs, aa=z.yh, home=0))])
        for t, x in long.groupby('t'):
            x = x.sort_values('d')
            teams.append(dict(sp=sp, lg=LEAGUE.get(t) if sp == 'Soccer' else None, t=t, g=int(len(x)), pf=round(float(x.pf.mean()), 2), af=round(float(x.af.mean()), 2), pa=round(float(x.pa.mean()), 2),
                              aa=round(float(x.aa.mean()), 2), of=round(float((x.af - x.pf).mean()), 2), oa=round(float((x.aa - x.pa).mean()), 2),
                              s=[[d, o, round(float(a), 1), int(b), round(float(c), 1), int(e)] for d, o, a, b, c, e in zip(x.d, x.o, x.pf, x.af, x.pa, x.aa)][-17:]))
    # every graded game's total: [sport, day, home, away, our projected total, actual total, market total or None, soccer league or None]
    totals = [[h['sp'], h['d'], h['h'], h['a'], round(h['hs'] + h['as'], 1), int(h['yh'] + h['ya']),
               (h.get('ln') or {}).get('T'), LEAGUE.get(h['h']) if h['sp'] == 'Soccer' else None] for h in H_rows]
    # games with a book line: [sport, day, home, away, our home margin, book home margin (-spread), actual home margin, our total, book total, actual total]
    lines = [[h['sp'], h['d'], h['h'], h['a'], round(h['hs'] - h['as'], 1), -float(h['ln']['L']), int(h['yh'] - h['ya']), round(h['hs'] + h['as'], 1),
              (h['ln'] or {}).get('T'), int(h['yh'] + h['ya'])] for h in H_rows if (h.get('ln') or {}).get('L') is not None]
    trend = []
    for (sp, d), x in X.groupby(['sp', 'd']):
        L = x[x.lm.notna()]
        trend.append(dict(sp=sp, d=d, n=int(len(x)), mae_margin=round(float(x.em.mean()), 2), mae_margin_line=round(float(L.lm.mean()), 2) if len(L) else None,
                          win_acc=round(float(x.win.dropna().mean()), 4) if x.win.notna().any() else None))
    return teams, trend, totals, lines

# ---- matchup buckets (J 2026-10-18, item 3 option 1+2): does our miss depend on how soft or tough the opponent is? ----
def box_long(sp, box):
    """one row per player-game: gid, date, team, opp (the defence he faced), grp (position group), names, stats in props keys"""
    if box is None or not len(box): return None
    if sp == 'NFL':
        b = box[box.season_type == 'REG'] if 'season_type' in box else box
        L = pd.DataFrame(dict(gid=b.game_id.astype(str), date=b.week, team=b.team, opp=b.opponent_team, grp=b.position.fillna('').str[:2],
                              n1=b.player_display_name.astype(str), n2=b.player_name.astype(str),
                              pass_att=b.attempts, cmp=b.completions, pass_yds=b.passing_yards, pass_td=b.passing_tds, carries=b.carries,
                              rush_yds=b.rushing_yards, rec=b.receptions, targets=b.targets, rec_yds=b.receiving_yards,
                              rush_rec_yds=b.rushing_yards.fillna(0) + b.receiving_yards.fillna(0), td=b.rushing_tds.fillna(0) + b.receiving_tds.fillna(0)))
        return L
    if sp == 'NHL':
        b = box.copy(); gk = ~b.position.isin(['C', 'L', 'R', 'D'])
        b = b[~gk | b.starter.fillna(False).astype(bool)]; gk = ~b.position.isin(['C', 'L', 'R', 'D'])
        tm = b.groupby('game_id').team_abbrev.unique().to_dict()
        opp = [next((t for t in tm[g] if t != x), None) for g, x in zip(b.game_id, b.team_abbrev)]
        nm = b.player_name.astype(str) + np.where(gk, ' (G)', '')
        return pd.DataFrame(dict(gid=b.game_id.astype(str), date=b.game_date, team=b.team_abbrev, opp=opp, grp=np.where(gk, 'G', np.where(b.position == 'D', 'D', 'F')),
                                 n1=nm, n2=nm, sog=np.where(gk, np.nan, b.shots_on_goal), points=np.where(gk, np.nan, b.points), goals=np.where(gk, np.nan, b.goals),
                                 assists=np.where(gk, np.nan, b.assists), blk=np.where(gk, np.nan, b.blocked_shots), hits=np.where(gk, np.nan, b.hits),
                                 saves=np.where(gk, b.saves, np.nan)))
    b = box[box.minutes.notna()] if 'minutes' in box else box
    c = lambda k: pd.to_numeric(b[k], errors='coerce') if k in b else np.nan
    pos = b.athlete_position_abbreviation.fillna('').str[-1:].replace({'': 'F'}) if 'athlete_position_abbreviation' in b else 'F'
    return pd.DataFrame(dict(gid=b.game_date.astype(str).str[:10], date=b.game_date.astype(str), team=b.team_abbreviation.astype(str), opp=b.opponent_team_abbreviation.astype(str),
                             grp=pos, n1=b.athlete_display_name.astype(str), n2=b.athlete_display_name.astype(str), pts=c('points'), reb=c('rebounds'), ast=c('assists'),
                             fg3m=c('three_point_field_goals_made'), stl=c('steals'), blk=c('blocks'), tov=c('turnovers'),
                             pra=c('points').fillna(0) + c('rebounds').fillna(0) + c('assists').fillna(0)))

def opp_index(L, stats):
    """{(gid, defence, grp, stat): what that defence allowed to the position group in its EARLIER games / league average}; needs 3+ earlier games"""
    out = {}
    if L is None: return out
    F = L.groupby(['gid', 'date', 'opp', 'grp'])[stats].sum(min_count=1).reset_index().sort_values('date')
    for s in stats:
        lg = F.groupby('grp')[s].mean()
        for (o, gp), x in F.groupby(['opp', 'grp']):
            pr = x[s].shift(1).expanding().mean(); k = x[s].shift(1).expanding().count()
            for gid, v, n in zip(x.gid, pr, k):
                if n >= 3 and v == v and lg.get(gp, 0) > 0: out[(gid, o, gp, s)] = v / lg[gp]
    return out

TIERS = ['tough', 'average', 'soft']

def tier_table(rows, keys):
    """rows have tier, y, m -> per keys+tier: n, actual-projected, % over projection, its standard error"""
    X = pd.DataFrame(rows); out = []
    if not len(X): return out
    for k, x in X.groupby(keys + ['tier']):
        k = k if isinstance(k, tuple) else (k,)
        d = x.y - x.m; r = dict(zip(keys + ['tier'], k))
        r.update(n=int(len(x)), ov=round(float(d.mean()), 3), pct=round(float(x.y.sum() / x.m.sum() - 1), 4) if x.m.sum() > 0 else None,
                 se=round(float(d.std(ddof=1) / np.sqrt(len(x))), 3) if len(x) > 1 else None)
        out.append(r)
    return out

def matchup_props(sp, D, box):
    """our prop miss by opponent tier (thirds of what the defence allowed that position group before this game)"""
    if D is None or not len(D): return []
    L = box_long(sp, box)
    if L is None: return []
    stats = sorted(set(D.stat) & set(L.columns))
    idx = opp_index(L, stats); grp = {}
    for r in L.itertuples():
        grp[(r.gid, r.team, r.n1)] = r.grp; grp[(r.gid, r.team, r.n2)] = r.grp
    rows = []
    for x in D[D.o.notna()].itertuples():
        gp = grp.get((x.g, x.t, x.n)); v = idx.get((x.g, x.o, gp, x.stat)) if gp else None
        if v is not None: rows.append(dict(stat=x.stat, grp=gp, v=v, y=x.y, m=x.m))
    if not rows: return []
    X = pd.DataFrame(rows)
    X['tier'] = X.groupby('stat').v.transform(lambda v: pd.qcut(v.rank(method='first'), 3, labels=TIERS) if len(v) >= 30 else pd.Series([None] * len(v), index=v.index))
    X = X[X.tier.notna()]; X['tier'] = X.tier.astype(str)
    return [dict(sp=sp, **r) for r in tier_table(X.to_dict('records'), ['stat'])]

def game_meta(sp):
    """finished games with dates and source team ids: gid, date, home, away, hs, as"""
    if sp == 'MLB':
        F = json.load(gzip.open(MLB_FIN, 'rt')) if os.path.exists(MLB_FIN) else {}
        if not F: return None
        M = pd.DataFrame(list(F.values()))
        return pd.DataFrame(dict(gid=M.gamePk.astype(str), date=M.date.astype(str), home=M.home, away=M.away, hs=M.homeRuns.astype(float), as_=M.awayRuns.astype(float)))
    try:
        with urllib.request.urlopen(GAME_SRC[sp], timeout=90) as r: b = r.read()
    except Exception: return None
    if sp == 'NFL':
        g = pd.read_csv(io.BytesIO(b)); g = g[g.home_score.notna() & (g.season == SEASON_NFL)]
        return pd.DataFrame(dict(gid=g.game_id.astype(str), date=g.gameday.astype(str), home=g.home_team, away=g.away_team, hs=g.home_score, as_=g.away_score))
    g = pd.read_parquet(io.BytesIO(b)); done = g.status.eq('STATUS_FINAL') if 'status' in g else g.status_type_completed.astype(bool); g = g[done]
    return pd.DataFrame(dict(gid=g.game_id.astype(str), date=g.game_date.astype(str), home=g.home_id.astype(str), away=g.away_id.astype(str), hs=g.home_score.astype(float), as_=g.away_score.astype(float)))

def matchup_games(G):
    """team score miss (actual - our projection) by the opponent's defence, offence and pace tier from its EARLIER games this season"""
    X = pd.DataFrame(G)
    if not len(X): return []
    out = []
    for sp, z in X.groupby('sp'):
        M = game_meta(sp) if sp in GAME_SRC or sp == 'MLB' else None
        if M is None or not len(M):    # no schedule file: the graded games themselves (NHL, soccer: every game is graded)
            M = pd.DataFrame(dict(gid=z.id.astype(str), date=z.d.astype(str), home=z.h, away=z.a, hs=z.yh, as_=z.ya))
        T = pd.concat([pd.DataFrame(dict(gid=M.gid, date=M.date, t=M.home, pf=M.hs, pa=M.as_)), pd.DataFrame(dict(gid=M.gid, date=M.date, t=M.away, pf=M.as_, pa=M.hs))]).sort_values('date')
        T['tot'] = T.pf + T.pa
        for c in ['pf', 'pa', 'tot']:
            T[c + '_pr'] = T.groupby('t')[c].transform(lambda v: v.shift(1).expanding().mean())
        T['k'] = T.groupby('t').cumcount()
        T = T[T.k >= 2].set_index(['gid', 't'])
        side = M.set_index('gid')
        rows = []
        for h in z.to_dict('records'):
            gid = str(h['id'])
            if gid not in side.index: continue
            hm, aw = side.loc[gid, 'home'], side.loc[gid, 'away']
            for opp, pf, af in ((aw, h['hs'], h['yh']), (hm, h['as'], h['ya'])):
                if (gid, opp) not in T.index: continue
                o = T.loc[(gid, opp)]
                rows.append(dict(m=pf, y=af, defn=o.pa_pr, off=o.pf_pr, pace=o.tot_pr))
        if len(rows) < 30: continue
        R = pd.DataFrame(rows)
        for dim, lab in (('defn', ['tough', 'average', 'soft']), ('off', ['weak', 'average', 'strong']), ('pace', ['slow', 'average', 'fast'])):
            R['tier'] = pd.qcut(R[dim].rank(method='first'), 3, labels=lab).astype(str)
            out += [dict(sp=sp, dim=dim, **r) for r in tier_table(R[['tier', 'y', 'm']].to_dict('records'), [])]
    return out

def shadow_grade(D):
    """NFL scheme shadow (models/nfl-player/shadow.py): candidate terms' misses next to the official projection's, same games"""
    f = f'{HIST}/NFL_SHADOW.json.gz'
    if D is None or not len(D) or not os.path.exists(f): return []
    S = json.load(gzip.open(f, 'rt')); rows = []
    for x in D.itertuples():
        v = S.get(f'{x.g}|{x.t}|{x.n}|{x.stat}')
        if v: rows.append(dict(stat=x.stat, y=x.y, m=x.m, cov=v['cov'], pers=v['pers']))
    if not rows: return []
    X = pd.DataFrame(rows); out = []
    for st, z in list(X.groupby('stat')) + [('all', X)]:
        e = {k: (z[k] - z.y).abs() for k in ['m', 'cov', 'pers']}
        out.append(dict(stat=st, n=int(len(z)), mae=round(float(e['m'].mean()), 3), mae_cov=round(float(e['cov'].mean()), 3), mae_pers=round(float(e['pers'].mean()), 3),
                        win_cov=round(float((e['cov'] < e['m']).sum() / max(1, (e['cov'] != e['m']).sum())), 3),
                        win_pers=round(float((e['pers'] < e['m']).sum() / max(1, (e['pers'] != e['m']).sum())), 3)))
    return out

# ---- sports without a game snapshot (J 2026-10-19: every chart for every sport) ----
def game_row(sp, gid, d, h, a, hs, as_, yh, ya):
    return dict(sp=sp, id=str(gid), d=d, k=d, h=h, a=a, hs=float(hs), yh=float(yh), ya=float(ya), ln={}, e=None, lm=None, lt=None, lwin=None, ew=None,
                em=abs((hs - as_) - (yh - ya)), et=abs((hs + as_) - (yh + ya)), es=(abs(hs - yh) + abs(as_ - ya)) / 2,
                win=None if yh == ya else int((hs > as_) == (yh > ya)), **{'as': float(as_)})

def nhl_games(D):
    """NHL has no game model: a team's projected goals = the sum of its skaters' goal projections (the same props the app
    shows), actual = their goals. Every NHL game has props, so this covers the whole slate."""
    if D is None or not len(D): return []
    X = D[(D.stat == 'goals')]
    if not len(X): return []
    T = X.groupby(['g', 't']).agg(d=('d', 'first'), o=('o', 'first'), m=('m', 'sum'), y=('y', 'sum'), h=('h', 'first') if 'h' in X else ('d', 'first')).reset_index()
    out = []
    for g, z in T.groupby('g'):
        if len(z) != 2: continue
        z = z.sort_values('h', ascending=False) if 'h' in X else z
        hr, ar = z.iloc[0], z.iloc[1]
        out.append(game_row('NHL', g, str(hr.d), hr.t, ar.t, hr.m, ar.m, hr.y, ar.y))
    return out

SOC = {'E0': 'en.1', 'SP1': 'es.1', 'D1': 'de.1', 'I1': 'it.1', 'F1': 'fr.1'}
SOC_ALIAS = {'manchester city': 'man city', 'manchester united': 'man united', 'nottingham forest': "nott'm forest", 'brighton & hove albion': 'brighton',
             'tottenham hotspur': 'tottenham', 'newcastle united': 'newcastle', 'leeds united': 'leeds', 'ipswich town': 'ipswich', 'hull city': 'hull',
             'coventry city': 'coventry', 'wolverhampton wanderers': 'wolves', 'west ham united': 'west ham', 'leicester city': 'leicester',
             'athletic club': 'ath bilbao', 'club atletico de madrid': 'ath madrid', 'atletico madrid': 'ath madrid', 'borussia monchengladbach': "m'gladbach",
             'internazionale milano': 'inter', 'paris saint-germain': 'paris sg', 'stade rennais 1901': 'rennes', 'real betis balompie': 'betis',
             'bayern munchen': 'bayern munich', 'borussia dortmund': 'dortmund', 'eintracht frankfurt': 'ein frankfurt', 'koln': 'fc koln',
             'bayer 04 leverkusen': 'leverkusen', 'rcd espanyol de barcelona': 'espanol', 'espanyol de barcelona': 'espanol', 'rayo vallecano de madrid': 'vallecano',
             'real sociedad de futbol': 'sociedad', 'paris': 'paris fc', 'espanyol de barcelona': 'espanol', 'real sociedad': 'sociedad', 'deportivo la coruna': 'la coruna', 'racing santander': 'santander'}

def soc_name(n, teams):
    """openfootball name -> the model's team name (exact after cleaning, then alias, then closest match >= 0.75)"""
    import difflib, unicodedata
    c = unicodedata.normalize('NFKD', n).encode('ascii', 'ignore').decode().lower()
    drop = {'fc', 'afc', 'cf', 'sc', 'ac', 'as', 'ssc', 'rc', 'ud', 'cd', 'sd', 'ca', 'rcd', '1.', 'vfb', 'vfl', 'tsg', 'sv', 'ss'}
    c = ' '.join(w for w in c.split() if w not in drop)
    c = SOC_ALIAS.get(c, c)
    low = {unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode().lower(): t for t in teams}
    if c in low: return low[c]
    best = difflib.get_close_matches(c, list(low), n=1, cutoff=0.75)
    if best: return low[best[0]]
    hit = [t for k, t in low.items() if k in c or c in k]
    return hit[0] if len(hit) == 1 else None

def soccer_games():
    """soccer (big 5): our model's expected goals (app/data/soccer_match.json lh/la, unblended) for every match played after
    the ratings' data cut-off (`through`), so every graded match is out of sample; results from openfootball on GitHub."""
    f = f'{APP}/soccer_match.json'
    if not os.path.exists(f): return []
    S = json.load(open(f)); out = []; miss = set()
    for code, L in S['leagues'].items():
        if code not in SOC: continue
        try:
            with urllib.request.urlopen(f'https://raw.githubusercontent.com/openfootball/football.json/master/2026-27/{SOC[code]}.json', timeout=60) as r: M = json.load(r)['matches']
        except Exception as e:
            print('Soccer', code, 'results not available:', e); continue
        for x in M:
            sc = x.get('score'); ft = sc.get('ft') if isinstance(sc, dict) else None
            if not ft or x['date'] <= L.get('through', ''): continue
            h, a = soc_name(x['team1'], L['teams']), soc_name(x['team2'], L['teams'])
            if not h or not a:
                miss.update(n for n, m in ((x['team1'], h), (x['team2'], a)) if not m); continue
            i, j = L['teams'].index(h), L['teams'].index(a); LEAGUE[h] = LEAGUE[a] = L['name']
            out.append(game_row('Soccer', f"{code}|{x['date']}|{h}|{a}", x['date'], h, a, L['lh'][i][j], L['la'][i][j], ft[0], ft[1]))
    if miss: print('Soccer names not matched:', sorted(miss))
    return out

def summary_row(sp, rows, model):
    X = pd.DataFrame(rows)
    return dict(sport=sp, model=model, n=int(len(X)), mae_margin=round(float(X.em.mean()), 2), mae_total=round(float(X.et.mean()), 2), mae_team=round(float(X.es.mean()), 2),
                slope=round(float(np.polyfit(X.hs - X['as'], X.yh - X.ya, 1)[0]), 3) if len(X) >= 20 else None,
                win_acc=round(float(X.win.dropna().mean()), 4) if X.win.notna().any() else None, n_line=0, mae_margin_line=None, mae_total_line=None,
                win_acc_line=None, edges=dict(n=0, w=0, l=0, p=0, units=0.0))

GAP_BANDS = {'NFL': [0, 1, 2, 3, 5], 'CFB': [0, 3, 5, 7, 10], 'NBA': [0, 2, 4, 6, 8], 'CBB': [0, 2, 4, 6, 8]}
def gap_records(G):
    """J 2026-10-26 (season-adjustments card 3): this season's record of our model's side against the line by how far our
    margin sits from the line, for the game cards' big-disagreement badge. Rows: [sport, gap at least, won, lost, pushed]."""
    out = []
    for sp, bands in GAP_BANDS.items():
        R = [g for g in G if g.get('sp') == sp and (g.get('ln') or {}).get('L') is not None]
        for lo in bands:
            w = l = pu = 0
            for g in R:
                gap = (g['hs'] - g['as']) + g['ln']['L']          # our margin minus the line's expected home margin (-L)
                if abs(gap) < lo or gap == 0: continue
                v = ((g['yh'] - g['ya']) + g['ln']['L']) * np.sign(gap)
                if v > 0: w += 1
                elif v < 0: l += 1
                else: pu += 1
            out.append([sp, lo, w, l, pu])
    return out

def nfl_home(G):
    """J 2026-10-26 (season-adjustments card 10, watch only): NFL home field this season. Home teams' average margin vs the
    closing line (nflverse games.csv) and vs our model's snapshots, home ATS, a t-stat, and the alert rule: home teams
    more than 2 pts worse than the line with t below -2 means it is time to re-estimate home field (else revisit week 12)."""
    try:
        with urllib.request.urlopen(GAME_SRC['NFL'], timeout=90) as r: g = pd.read_csv(io.BytesIO(r.read()))
    except Exception as e:
        print('NFL home-field tracker: games not available:', e); return None
    g = g[(g.season == SEASON_NFL) & g.home_score.notna() & (g.location != 'Neutral') & (g.game_type == 'REG')]
    if not len(g): return None
    m = g.home_score - g.away_score; ats = m - g.spread_line
    ours = {str(x['id']): x['hs'] - x['as'] for x in G if x.get('sp') == 'NFL'}
    ok = g.game_id.astype(str).isin(ours)
    mod = g.game_id.astype(str).map(ours)
    wk = [[int(w), round(float(m[g.week == w].mean()), 2), round(float(g.spread_line[g.week == w].mean()), 2)] for w in sorted(g.week.unique())]
    t = float(ats.mean() / (ats.std() / np.sqrt(len(ats)))) if len(ats) > 2 else None
    lean = float((mod[ok] - g.spread_line[ok]).mean()) if ok.any() else None   # our model's home lean beyond the line
    return dict(n=int(len(g)), home_margin=round(float(m.mean()), 2), line=round(float(g.spread_line.mean()), 2), ats_w=int((ats > 0).sum()), ats_l=int((ats < 0).sum()),
                t=round(t, 2) if t is not None else None, n_model=int(ok.sum()), model=round(float(mod[ok].mean()), 2) if ok.any() else None,
                model_actual=round(float(m[ok].mean()), 2) if ok.any() else None, lean=round(lean, 2) if lean is not None else None,
                alert=bool(ats.mean() < -2 and t is not None and t < -2), vs_line=round(float(ats.mean()), 2), weeks=wk, normal=dict(home_margin=2.2, line=1.6, note='2021-25 averages'))

# ---------- line-based prop flags (Bet Desk v71, J 2026-10-27): grade snap_flags.js rows at the flagged line and price ----------
FLAGS, FLAGS_CLOSE = f'{HIST}/FLAGS.json.gz', f'{HIST}/FLAGS_CLOSE.json.gz'
PO_MAP = {'NFL': {'Pass Yards': 'pass_yds', 'Pass Attempts': 'pass_att', 'Pass Completions': 'cmp', 'TD Passes': 'pass_td', 'Rush Yards': 'rush_yds',
                  'Rush Attempts': 'carries', 'Receiving Yards': 'rec_yds', 'Receptions': 'rec', 'To Score TD': 'td'},
          'NBA': {'Points': 'pts', 'Rebounds': 'reb', 'Assists': 'ast', '3 Point FG': 'fg3m', 'Points + Assists + Rebounds': 'pra', 'Steals': 'stl', 'Blocks': 'blk', 'Turnovers': 'tov'},
          'NHL': {'Shots On Goal': 'sog', 'Points': 'points', 'Goals': 'goals', 'Anytime Goal Scorer': 'goals', 'Assists': 'assists', 'Blocks': 'blk', 'Saves': 'saves'}}   # = PO_MAP in bet-desk.html

def _imp(a): a = float(a); return 100 / (a + 100) if a > 0 else -a / (-a + 100)
def _dec(a): a = float(a); return 1 + (a / 100 if a > 0 else 100 / -a)
def _pnorm(n):
    import unicodedata, re
    n = unicodedata.normalize('NFD', str(n).lower()); n = ''.join(c for c in n if not unicodedata.combining(c))
    n = re.sub(r"[.'’,-]", ' ', n); n = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', '', n); w = n.replace('(g)', '').split()
    return (w[0][0] + ' ' + ' '.join(w[1:])) if len(w) > 1 else (w[0] if w else '')

def capture_close():
    """Before each flagged game starts, keep the newest price at the flagged book/line/side and the no-vig market at that line
    (median over books with both sides) from app/data/odds/props_{sp}.json. The last sweep before kick = the close we have
    (the odds pull runs once a day, so this is the closest pregame price, not the true closing line)."""
    if not os.path.exists(FLAGS): return {}
    F = json.load(gzip.open(FLAGS, 'rt')); C = json.load(gzip.open(FLAGS_CLOSE, 'rt')) if os.path.exists(FLAGS_CLOSE) else {}
    now = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'); n = 0
    for sp in PO_MAP:
        f = f'{APP}/odds/props_{sp}.json'
        if not os.path.exists(f): continue
        O = json.load(open(f)); sweep = O.get('made', '')
        idx = {}
        for g in O.get('games', []):
            if str(g.get('k', '')) <= now: continue                      # started: its prices are no longer pregame
            for nm, mk in g.get('players', {}).items(): idx[(g['k'], _pnorm(nm))] = mk
        for key, r in F.items():
            if r['sp'] != sp or str(r.get('kick', '')) <= now: continue
            mk = idx.get((r['kick'], _pnorm(r['n'])))
            if not mk: continue
            for mname, st in PO_MAP[sp].items():
                if st != r['k'] or mname not in mk: continue
                yes = mname in ('To Score TD', 'Anytime Goal Scorer')
                L = next((v for l, v in mk[mname].items() if yes or abs(float(l) - r['line']) < 1e-6), None)
                if not L: continue
                side = ('Yes' if r['side'] == 'Over' else 'No') if yes else r['side']; other = {'Over': 'Under', 'Under': 'Over', 'Yes': 'No', 'No': 'Yes'}[side]
                px = (L.get(r['book']) or {}).get(side)
                fair = [_imp(b[side]) / (_imp(b[side]) + _imp(b[other])) for b in L.values() if side in b and other in b]
                if px is None and not fair: continue
                C[key] = dict(close=px, fair=round(float(np.median(fair)), 4) if fair else None, nb=len(fair), sweep=sweep); n += 1; break
    json.dump(C, gzip.open(FLAGS_CLOSE, 'wt')); print('line flags: close prices kept for', n, 'flags, history', len(C))
    return C

def grade_line_flags(RES, C):
    """scorecard `lineflags`: every snap_flags.js row with a box score, won/lost/pushed at the flagged line, profit per 1 unit
    at the flagged price, and CLV vs the last pregame sweep: clv = flagged price's payout x closing no-vig chance - 1
    (above 0 = we got a better price than the closing market said it was worth), beat = flagged price better than the same
    book's close (+1 better, 0 same, -1 worse). CLV only counts when a sweep newer than the flag's own sweep reached us before kick. A flagged player who did not play (no box row) is void, as at the books."""
    if not os.path.exists(FLAGS): return {}
    F = json.load(gzip.open(FLAGS, 'rt')); rows = []
    for key, r in F.items():
        sp, res = r['sp'], RES.get(r['sp'], {})
        n = r['n'] + (' (G)' if sp == 'NHL' and r['k'] == 'saves' and not r['n'].endswith('(G)') else '')
        v = res.get(((r.get('d') or '')[:10] if sp == 'NBA' else str(r['g']), r['t'], n), {}).get(r['k'])
        if v is None: continue
        res_ = 'P' if v == r['line'] else 'W' if (v > r['line']) == (r['side'] == 'Over') else 'L'
        u = {'W': _dec(r['price']) - 1, 'L': -1.0, 'P': 0.0}[res_]
        c = C.get(key, {})
        if str(c.get('sweep', '')) <= str(r.get('sweep', '')): c = {}     # no sweep after the flag's own one yet: no CLV
        clv = round(_dec(r['price']) * c['fair'] - 1, 4) if c.get('fair') else None
        beat = None if c.get('close') is None else (_dec(r['price']) > _dec(c['close']) + 1e-9) - (_dec(r['price']) < _dec(c['close']) - 1e-9)
        rows.append(dict(sp=sp, kind=r['kind'], n=r['n'], t=r['t'], o=r.get('o'), d=(r.get('d') or r.get('kick', ''))[:10], k=r['k'], side=r['side'], line=r['line'],
                         book=r['book'], price=r['price'], ev=r.get('ev'), evb=r.get('evb'), proj=r.get('proj'), act=v, res=res_, u=round(u, 3),
                         close=c.get('close'), cfair=c.get('fair'), clv=clv, beat=beat))
    def agg(R):
        w, l, p = (sum(x['res'] == s for x in R) for s in 'WLP'); u = sum(x['u'] for x in R); risk = w + l
        cl = [x['clv'] for x in R if x['clv'] is not None]; bt = [x['beat'] for x in R if x['beat'] is not None]
        return dict(n=len(R), w=w, l=l, p=p, units=round(u, 2), roi=round(u / risk, 4) if risk else None,
                    ev=round(float(np.mean([x['ev'] for x in R if x['ev'] is not None])), 4) if R else None,
                    clv=round(float(np.mean(cl)), 4) if cl else None, nclv=len(cl), beat=round(sum(b > 0 for b in bt) / len(bt), 3) if bt else None)
    out = dict(made=dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'), pending=len(F) - len(rows), all=agg(rows), by=[])
    for sp in sorted({x['sp'] for x in rows}):
        for kd in ['all', 'flag', 'proven', 'check']:
            R = [x for x in rows if x['sp'] == sp and (kd == 'all' or x['kind'] == kd)]
            if R: out['by'].append(dict(sp=sp, kind=kd, **agg(R)))
    for kd in ['flag', 'proven', 'check']:
        R = [x for x in rows if x['kind'] == kd]
        if R: out['by'].append(dict(sp='All', kind=kd, **agg(R)))
    days = sorted({x['d'] for x in rows}); cum = 0; out['daily'] = []
    for d in days:
        R = [x for x in rows if x['d'] == d and x['kind'] != 'check']; cum += sum(x['u'] for x in R)
        out['daily'].append(dict(d=d, n=len(R), units=round(sum(x['u'] for x in R), 2), cum=round(cum, 2)))
    out['rows'] = sorted(rows, key=lambda x: x['d'], reverse=True)[:300]
    print('line flags graded', len(rows), 'waiting', out['pending'], out['all'])
    return out

if __name__ == '__main__':
    NHLG = []
    SC = dict(made=dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC'), props=[], calib=[], flags=[], games=[], counts={}, through={}, team_props=[], trend=[], matchup=[], play_min=PLAY_MIN); PR, TRK, RES = [], [], {}
    for sp in ['NFL', 'NHL', 'NBA']:
        box = fetch(sp); res = results(sp, box); RES[sp] = res
        snapshot(sp, res)
        p, c, f, n, thr, pr, D = grade(sp, res)
        SC['props'] += p; SC['calib'] += c; SC['flags'] += f; SC['counts'][sp] = n; PR += pr
        tp, tt, tr = track(sp, D); TRK += tp; SC['team_props'] += tt; SC['trend'] += tr
        SC['matchup'] += matchup_props(sp, D, box)
        if sp == 'NFL': SC['shadow'] = shadow_grade(D)
        if sp == 'NHL': NHLG = nhl_games(D)
        if thr: SC['through'][sp] = thr
        print(sp, 'graded', n)
    snapshot_mlb(); snapshot_cbb_shadow()
    SC['games'], SC['recent'], SC['gcounts'], SC['pending'], G = grade_games()
    for sp, rows, model in (('NHL', NHLG, 'From player props'), ('Soccer', soccer_games(), 'Bet Desk')):
        if rows: SC['games'].append(summary_row(sp, rows, model)); SC['gcounts'][sp] = len(rows); G += rows
    SC['teams'], SC['gtrend'], SC['gtotals'], SC['glines'] = track_games(G)
    SC['gmatchup'] = matchup_games(G)
    SC['gapsize'] = gap_records(G); SC['nfl_home'] = nfl_home(G)
    SC['lineflags'] = grade_line_flags(RES, capture_close())
    print('games graded', SC['gcounts'], 'waiting', SC['pending'])
    json.dump(SC, open(f'{APP}/scorecard.json', 'w'), separators=(',', ':'), default=float)
    json.dump(dict(made=SC['made'], rows=PR, track=TRK), open(f'{APP}/player_results.json', 'w'), separators=(',', ':'), default=float)
    print('player results', len(PR))
    print('wrote', f'{APP}/scorecard.json')
