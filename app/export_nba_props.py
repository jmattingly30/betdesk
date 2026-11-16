"""Export the NBA player prop model's calibrated over/under curves for one game day, for the Bet Desk app.

python3 export_nba_props.py DATE [out.json] [--games overrides.csv] [--dl dir] [--days N]

DATE is the US game day (YYYY-MM-DD). Every team's NEXT game on or after DATE (within N days, default 7) is projected,
so the slate covers all 30 teams (2026-10-15, J: "only about 8 teams"). Last season's playoff games are left out of the
history while a new season has fewer than 5 games per team, so playoff rotations don't set opening-night minutes. Games come from ESPN's 2026-27 schedule and rosters from ESPN's current team
rosters (both mirrored by sportsdataverse on GitHub, downloaded into --dl, default /tmp/nba27). Offseason moves are
handled automatically: anyone on a team's latest box-score roster who is no longer on its current roster is dropped,
and anyone who joined is added. Rookies with no NBA games are skipped (the model needs history).

Spread and total for each game: from --games if given, otherwise from the NBA team model (models/nba-team/predict.py)
with the same roster moves. Market numbers are better; enter them when you have them.
overrides.csv: team,spread,total,out,q   (team = ESPN code; spread = that team's expected margin, + favoured;
out = players not playing, ';'-separated; q = 'Name:0.6;Name2:0.25' questionable with chance of playing).
Any column may be blank. The injury report is not fetched (ESPN's API is blocked from the cloud), so list outs here.

Runs models/nba-player/project.py unchanged, then writes P(over) at every half-point line across each player's range
from the same calibrated distribution project.py uses."""
import sys, os, re, json, runpy, datetime, subprocess, unicodedata, urllib.request
import numpy as np, pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
M = os.path.normpath(f'{here}/../models/nba-player'); T = os.path.normpath(f'{here}/../models/nba-team')
args = sys.argv[1:]
def opt(name, default=None):
    if name in args:
        k = args.index(name); v = args[k + 1]; del args[k:k + 2]; return v
    return default
ov_path, dl, ndays = opt('--games'), opt('--dl', '/tmp/nba27'), int(opt('--days', '7'))
date = args[0]; out = args[1] if len(args) > 1 else f'{here}/data/nba_props.json'
os.makedirs(dl, exist_ok=True)
SRC = {'sched.parquet': 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_schedules/nba_schedule_2027.parquet',
       'rosters.parquet': 'https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba/rosters/parquet/rosters_2027.parquet'}
for f, u in SRC.items():   # refresh every run: rosters change daily in season
    urllib.request.urlretrieve(u, f'{dl}/{f}')

def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', re.sub(r'[^a-z ]', '', s)).split())

sch = pd.read_parquet(f'{dl}/sched.parquet')
sch['day'] = pd.to_datetime(sch.date, utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
end = (pd.Timestamp(date) + pd.Timedelta(days=ndays)).strftime('%Y-%m-%d')
fut = sch[(sch.day >= date) & (sch.day <= end)].sort_values('date')
covered, keep, export_for = set(), [], {}
for i, r in fut.iterrows():
    need = [t for t in (r.home_abbreviation, r.away_abbreviation) if t not in covered]
    if need: keep.append(i); export_for[r.id] = set(need); covered |= set(need)
day = fut.loc[keep]
if day.empty: sys.exit(f'no NBA games from {date} to {end}')
print(f'{len(day)} games, {len(covered)} teams, {day.day.min()} to {day.day.max()}')
ros = pd.read_parquet(f'{dl}/rosters.parquet'); ros['pid'] = ros.athlete_id.astype('int64')
pg = pd.read_parquet(f'{M}/data/player_games.parquet', columns=['pid', 'name', 'team', 'season', 'date', 'played', 'playoff'])
new_season = (pg[pg.season == pg.season.max()].groupby('team').size().median() < 5) or (pd.Timestamp(date) - pg.date.max()).days > 60
# run the model on a copy of the model folder whose history drops last season's playoffs (start of a new season only)
import tempfile, shutil
MR = tempfile.mkdtemp(prefix='nbaM_')
for f in os.listdir(M):
    if f != 'data': os.symlink(f'{M}/{f}', f'{MR}/{f}')
os.makedirs(f'{MR}/data')
for f in os.listdir(f'{M}/data'):
    if f != 'player_games.parquet': os.symlink(f'{M}/data/{f}', f'{MR}/data/{f}')
full = pd.read_parquet(f'{M}/data/player_games.parquet')
if new_season:
    full = full[~((full.playoff == 1) & (full.season == full.season.max()))]; print('new season: last season playoff games left out of the history')
full.to_parquet(f'{MR}/data/player_games.parquet')
pg = pg[pg.index.isin(full.index)] if new_season else pg
last = pg[pg.played].sort_values('date').groupby('pid').tail(1).set_index('pid')
cur_team = dict(zip(ros.pid, ros.team_abbreviation))
ov = pd.read_csv(ov_path, dtype=str).fillna('').set_index('team') if ov_path else pd.DataFrame()

def moves(team):
    """(out, add) name lists turning the team's latest box-score roster into its current one."""
    lt = last[(last.team == team) & (last.season == last[last.team == team].season.max())]
    out = [n for p, n in zip(lt.index, lt.name) if cur_team.get(p) != team]
    add = [last.loc[p, 'name'] for p in ros[ros.team_abbreviation == team].pid if p in last.index and last.loc[p, 'team'] != team]
    return out, add

def ovget(team, col): return ov.loc[team, col] if team in ov.index and col in ov.columns else ''

rows, notes = [], []
for _, g in day.iterrows():
    h, a = g.home_abbreviation, g.away_abbreviation
    mv = {t: moves(t) for t in (h, a)}
    o = {t: ';'.join(mv[t][0] + [x for x in ovget(t, 'out').split(';') if x.strip()]) for t in (h, a)}
    sp, tot = ovget(h, 'spread'), ovget(h, 'total') or ovget(a, 'total')
    if not sp and ovget(a, 'spread'): sp = str(-float(ovget(a, 'spread')))
    src = 'market (entered)'
    if not (sp and tot):
        r = subprocess.run([sys.executable, 'predict.py', h, a, '--out-home', o[h], '--out-away', o[a],
                            '--add-home', ';'.join(mv[h][1]), '--add-away', ';'.join(mv[a][1])] +
                           (['--neutral'] if g.neutral_site else []), cwd=T, capture_output=True, text=True, check=True)
        x = re.search(r'^(?:projection|with absences)\s+\S+\s+\S+\s+(-?[\d.]+)\s+([\d.]+)', r.stdout, re.M)
        sp, tot, src = sp or x.group(1), tot or x.group(2), 'NBA team model' if not (sp or tot) else 'mixed'
    sp, tot = float(sp), float(tot)
    print(h, 'moves out/in:', mv[h], '|', a, mv[a])
    notes.append(f'{g.day[5:]} {a} @ {h}: home margin {sp:+.1f}, total {tot:.1f} ({src})')
    for t, opp, home, s in ((h, a, 1, sp), (a, h, 0, -sp)):
        rows.append(dict(date=g.day, gid=g.id, export=t in export_for[g.id], team=t, opp=opp, home=home, spread=s, total=tot, out=o[t], add=';'.join(mv[t][1]), q=ovget(t, 'q')))
print('\n'.join(notes))
gpath = f'{dl}/games_{date}.csv'; pd.DataFrame(rows).drop(columns=['gid', 'export']).to_csv(gpath, index=False)

proj = f'{dl}/proj_{date}.csv'
sys.argv = ['project.py', MR, gpath, '-', proj]
G = runpy.run_path(f'{M}/project.py', run_name='__main__')
shutil.rmtree(MR, ignore_errors=True)
fu, m = G['fu'], G['m']
# ---- Team minutes step (2026-11-16, docs/research/basketball.md idea 3) -------------------------------------------------
# A prop is void when the player sits, so every number shown is IF HE PLAYS. The model's own minutes are already that
# (it is trained on games played). What a team has to add up to is its EXPECTED minutes: sum of P(plays) x minutes-if-plays
# = 241. The old steps squeezed the whole 13-15 man list toward 241 first and divided by a P(plays) that averaged 0.90 when
# 78% of projected players actually played (Nov 5-15), so about 39 projected minutes a team sat with players who did not
# play and everyone who did was 1.5 to 2 minutes low, the bench most.
#   P(plays): how often a projected player has played, by whether he played his team's last game and how many of its last
#             five he played, counting every absence. Learned from our own logged projections (models/grading/history),
#             so it describes the players we actually project. Log loss .574 -> .413 on Nov 5-15.
#   Shortfall: the minutes freed by expected absences go to players in proportion to minutes x room below CEIL, so the bench
#             takes most and a 34-minute starter almost none.
# Walk-forward on Oct 28 - Nov 15 (table and CEIL from dates to Nov 4, scored on Nov 5-15, 1,572 player-games): minutes miss
# 5.08 -> 4.86, bias -1.49 -> +0.46; points bias -0.72 -> +0.15, points miss unchanged (4.63 vs 4.67).
# Scripts: project folder research/basketball-2026-11/ (nba_diag.py, nba_pplay.py, nba_bt.py, nba_eval.py).
CEIL = 36.0
P_TABLE = {(0, 0): .08, (0, 1): .33, (0, 2): .32, (0, 3): .43, (0, 4): .39, (1, 1): .62, (1, 2): .73, (1, 3): .91, (1, 4): .88, (1, 5): .95}   # used until 500 logged rows exist

def play_history(pgs, keys, day):
    """(played his team's last game, games played of its last five) for each (team, name key) before `day`, every absence counted."""
    pgs = pgs[pgs.date < pd.Timestamp(day)]; out = {}
    dates = pgs.groupby('team').date.apply(lambda x: sorted(x.unique())[-5:]).to_dict()
    pl = pgs[pgs.played].groupby(['team', 'key']).date.apply(set).to_dict()
    for t, k in keys:
        l5 = dates.get(t, []); mine = pl.get((t, k), set())
        out[(t, k)] = (int(bool(l5) and l5[-1] in mine), sum(d in mine for d in l5))
    return out

def p_plays_table(pgs):
    """P(plays) by (played last game, games of last five) from the logged projections; P_TABLE while the log is short."""
    hp = f'{here}/../models/grading/history/NBA.json.gz'
    try:
        import gzip
        H = pd.DataFrame(json.load(gzip.open(hp)).values()); H = H[(H.stat == 'pts') & H.we.notna() & (H.d < date)]
        H = H[H.d.isin(set(pgs.date.dt.strftime('%Y-%m-%d')))]
        if len(H) < 500: return dict(P_TABLE)
        H['key'] = H.n.map(norm); rows = []
        played = set(zip(pgs.date.dt.strftime('%Y-%m-%d')[pgs.played], pgs.team[pgs.played], pgs.key[pgs.played]))
        for d_, g in H.groupby('d'):
            ph = play_history(pgs, set(zip(g.t, g.key)), d_)
            rows += [(*ph[(t, k)], (d_, t, k) in played) for t, k in zip(g.t, g.key)]
        T = pd.DataFrame(rows, columns=['last', 'k5', 'y']); prior = T.groupby('k5').y.mean()
        tab = {(int(l), int(k)): float((g.y.sum() + 5 * prior[k]) / (len(g) + 5)) for (l, k), g in T.groupby(['last', 'k5'])}
        return {**P_TABLE, **tab}
    except Exception as e:
        print('P(plays) table: using the saved one (', e, ')'); return dict(P_TABLE)

def team_step(fu):
    cg = pd.read_parquet(f'{M}/data/player_games.parquet', columns=['name', 'team', 'season', 'date', 'played', 'playoff'])
    cg = cg[(cg.season == cg.season.max()) & (cg.playoff != 1) & (cg.date < pd.Timestamp(date))].copy(); cg['key'] = cg.name.map(norm)
    tab = p_plays_table(cg); fu = fu.copy(); fu['key'] = fu.name.map(norm)
    ph = play_history(cg, set(zip(fu.team, fu.key)), date)
    fu['p_plays'] = [min(0.98, max(0.03, tab.get(ph[(t, k)], 0.5))) for t, k in zip(fu.team, fu.key)]
    # project.py pulled every team halfway to 241 over its whole list; undo that to get the model's own minutes back
    k = fu.groupby(['game_id', 'team']).min_proj.transform('sum') / 241.0
    raw = fu.min_proj * k
    w = raw * (1 - raw / CEIL).clip(lower=0); g = [fu.game_id, fu.team]
    short = 241.0 - (fu.p_plays * raw).groupby(g).transform('sum'); den = (fu.p_plays * w).groupby(g).transform('sum')
    new = raw + (short / den.clip(lower=1e-6)).clip(-0.5, 1.5) * w
    f = new / fu.min_proj.clip(lower=0.5)
    fu['min_proj'] = new
    for c in [c for c in fu if c.startswith('mu_')]: fu[c] = fu[c] * f
    print('team minutes step: mean P(plays) %.3f | expected to play %.1f of %.1f projected per team | minutes if he plays, team sum %.0f' % (
        fu.p_plays.mean(), fu.groupby(g).p_plays.sum().mean(), fu.groupby(g).size().mean(), fu.groupby(g).min_proj.sum().mean()))
    return fu[fu.min_proj >= 3].drop(columns=['key']).copy()

def old_steps(fu):
    """The steps used before 2026-11-16. Still used at the start of a season, when there is no play history for the table."""
    # Preseason/early-season rosters are ESPN's full 15-18 man lists, so team minutes run well past 240, and project.py's
    # uniform halfway pull then cuts stars' minutes as much as the 15th man's. For a team over 250, undo that pull
    # (project.py multiplied everyone by 241 / current sum) and take the excess out of the bench (under 20 projected
    # minutes) first, then everyone, so starters keep their usual minutes. Means scale with minutes.
    def trim(g):
        f = np.full(len(g), g.min_proj.sum() / 241.0); raw = g.min_proj * f
        excess = raw.sum() - 241.0; bench = raw < 20
        if raw[bench].sum() >= excess: f[bench.values] *= (raw[bench].sum() - excess) / raw[bench].sum()
        else: f[bench.values] *= 0.05; f *= 241.0 / (raw * np.where(bench, 0.05, 1.0)).sum()
        g = g.copy(); g['min_proj'] *= f
        for c in [c for c in g if c.startswith('mu_')]: g[c] *= f
        return g
    big = fu.groupby(['game_id', 'team']).min_proj.transform('sum') > 250
    if big.any():
        print('roster bloat, bench minutes trimmed for:', sorted(set(fu.team[big])))
        fu.loc[big] = pd.concat([trim(g) for _, g in fu[big].groupby(['game_id', 'team'])])
    # Start of a season (2026-10-15 fix): with full 15-18 man ESPN rosters the model spreads minutes too evenly (starters
    # ~28 min, deep bench ~10), so opening-night minutes blend 70% the league's minutes-by-rotation-rank (2025-26 regular
    # season: 35.4, 33.2, 31.1, 28.9, 26.5, 23.3, 20.1, 17.0, 13.3, 9.1, 5.6, 4.4, 4.1 ...) with 30% the model, ranked by the
    # model's own minutes, then rescaled to 241. Stat means scale with minutes. In-season the model's minutes are used as is.
    RANK = np.array([35.4, 33.2, 31.1, 28.9, 26.5, 23.3, 20.1, 17.0, 13.3, 9.1, 5.6, 4.4, 4.1, 3.7, 3.2] + [2.0] * 10)
    def rank_minutes(g):
        g = g.sort_values('min_proj', ascending=False).copy(); n = len(g)
        t = RANK[:n] * 241.0 / RANK[:n].sum() if n >= 9 else RANK[:n]
        new = 0.7 * t + 0.3 * g.min_proj.values; new *= 241.0 / new.sum()
        f = new / g.min_proj.clip(lower=0.5).values
        g['min_proj'] = new
        for c in [c for c in g if c.startswith('mu_')]: g[c] *= f
        return g
    if new_season:
        fu = pd.concat([rank_minutes(g) for _, g in fu.groupby(['game_id', 'team'])]); print('new season: minutes set from rotation rank (70%) + model (30%)')
    # Early season (J 2026-10-26, season-adjustments card 4): new rotations aren't in the minutes model yet. Opening week
    # graded 2026-27: players who played got 2 more minutes than projected and 15% of projected players never dressed.
    # Through a team's first 15 games, blend each player's projected minutes with his minutes per available team game this
    # season (DNP-coach's decision counts as 0, injury absences don't count): weight g/(g+1), fading to 0 from game 10 to 15.
    # Checked on Oct 22-25 games (walk-forward): minutes error for players who played 5.31 -> 4.97, all players 5.75 -> 5.22.
    cg = pd.read_parquet(f'{M}/data/player_games.parquet', columns=['name', 'team', 'season', 'date', 'played', 'dnp_injury', 'min', 'playoff'])
    cg = cg[(cg.season == cg.season.max()) & (cg.playoff != 1) & (cg.date < pd.Timestamp(date))]
    if len(cg):
        cg['m0'] = cg['min'].fillna(0).where(cg.played, 0)
        av = cg[cg.played | ~cg.dnp_injury.fillna(False).astype(bool)]
        cur = av.groupby(['team', 'name']).agg(g=('m0', 'size'), mpg=('m0', 'mean'))
        tgames = cg.groupby('team').date.nunique()
        def season_minutes(g):
            t = g.team.iloc[0]; n = int(tgames.get(t, 0))
            if n == 0 or n >= 15: return g
            fade = 1.0 if n <= 10 else (15 - n) / 5
            x = cur.reindex(list(zip(g.team, g.name)))
            w = (x.g / (x.g + 1.0)).fillna(0).values * fade
            new = (1 - w) * g.min_proj.values + w * x.mpg.fillna(0).values
            new *= g.min_proj.sum() / max(new.sum(), 1e-6)
            f = new / g.min_proj.clip(lower=0.5).values
            g = g.copy(); g['min_proj'] = new
            for c in [c for c in g if c.startswith('mu_')]: g[c] *= f
            return g
        fu = pd.concat([season_minutes(g) for _, g in fu.groupby(['game_id', 'team'])]); print('early season: minutes blended with this season\'s minutes per game')
    fu = fu[fu.min_proj >= 3].copy()   # trimmed-out bench players
    # Props are void when a player sits, so show each player IF HE PLAYS (J 2026-10-28, OddsPapi thread). The minutes above
    # are averages that count coach's-decision DNPs as 0 (projected 20.0 vs 22.7 for players who played, Oct 24-26; bench
    # 6 -> 12). Divide minutes and means by P(plays) = (games played + 2 x 0.85) / (games available + 2) this season
    # (injury absences not counted), floor 0.5. Graded on Oct 16-26 games vs real Pinnacle/DK/FD prop prices: log loss
    # .7153 -> .7042 (models/oddspapi/props_backtest.py live). A flat x1.11 on everyone scored the same (.7021), so most of
    # the gain is that team minutes are normalised over players who then sit; recheck P(plays) vs a flat scale in December.
    if len(cg):
        avail = cg[cg.played | ~cg.dnp_injury.fillna(False).astype(bool)].groupby(['team', 'name']).agg(n=('played', 'size'), k=('played', 'sum'))
        x = avail.reindex(list(zip(fu.team, fu.name)))
        pplay = ((x.k.fillna(0) + 2 * 0.85) / (x.n.fillna(0) + 2)).clip(0.5, 1).values
    else:
        pplay = np.full(len(fu), 0.85)
    fu['min_proj'] = fu.min_proj / pplay
    for c in [c for c in fu if c.startswith('mu_')]: fu[c] = fu[c] / pplay
    print('props shown if the player plays: minutes / P(plays), mean P', round(float(np.mean(pplay)), 3))
    return fu

fu = old_steps(fu) if new_season else team_step(fu)
SHOW = ['pts', 'reb', 'ast', 'fg3m', 'pra', 'stl', 'blk', 'tov']
games = {}
players = []
for _, r in fu.iterrows():
    dstr = pd.Timestamp(r.date).strftime('%Y-%m-%d')
    gm = next(x for x in rows if x['team'] == r.team and x['date'] == dstr)
    if not gm['export']: continue          # this team's earlier game is already on the slate
    gid = games.setdefault((dstr, frozenset([r.team, r.opp])), len(games) + 1)
    st = {}
    for s in SHOW:
        k, pm = m.dist(r, s)
        c = np.cumsum(pm); q = [int(k[min(np.searchsorted(c, z), len(k) - 1)]) for z in (0.1, 0.5, 0.9)]
        hi = int(k[min(np.searchsorted(c, 0.995), len(k) - 1)])
        curve = []
        for L in np.arange(0.5, max(hi, 1) + 1, 1.0):
            p = float(np.clip(pm[k > L].sum(), 1e-4, 1 - 1e-4)); cc = getattr(m, 'temp', {}).get(s)
            if cc: p = 1 / (1 + np.exp(-(cc['intercept'] + cc['slope'] * np.log(p / (1 - p)))))
            curve.append([float(L), round(p, 4)])
        st[s] = dict(m=round(float(r['mu_' + s]), 1), q=q, c=curve)
    players.append(dict(n=r['name'], t=r.team, o=r.opp, d=dstr, p=f"{r.pos_grp} · {r.min_proj:.0f} min" if isinstance(r.pos_grp, str) else f'{r.min_proj:.0f} min',
                        g=gid, h=int(gm['home']), imp=round((gm['total'] + gm['spread']) / 2, 1), sp=round(gm['spread'], 1), s=st))
players.sort(key=lambda p: (p['g'], p['t'], -p['s']['pts']['m']))
doc = dict(sport='NBA', season=2027, week=f'{day.day.min()} to {day.day.max()}', made=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
           note='Assumes the player plays (props are void otherwise). Injuries only as entered; roster moves from ESPN rosters. '
                + ' | '.join(notes),
           games=sorted(games.values()), players=players)
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(doc, open(out, 'w'), separators=(',', ':'), allow_nan=False)
print(f'{len(players)} players, {len(games)} games -> {out} ({os.path.getsize(out) / 1024:.0f} KB)')
