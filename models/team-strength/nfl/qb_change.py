"""Backup-QB adjustment for the NFL team card (J 2026-10-18: PHI was rated as if Hurts were playing).
python3 qb_change.py <nflverse games.csv> [pbp dir]     # pbp dir default /tmp/claude-0/nfldl (pbp_YYYY.parquet)

The team ratings learn from final scores and closing lines, so they don't know who starts. nflverse games.csv names
each game's starting QB (for upcoming games, the projected starter). A team's usual starter = the QB with the most starts in its
previous 8 games. For every 2012-2025 game, this fits how far the final margin (and the closing line) moved from the
ratings' prediction when a team started someone else, split into the first such game ("new") and later games in a row
("cont"), since the market rating has already absorbed part of a QB change after one game.
QB value: each QB's EPA per dropback before the game (all earlier dropbacks since 2016, 0.7 weight per season back,
shrunk toward -0.05 with 250 dropbacks, about a replacement backup). The card's QB term = coef x (value of today's
starter - value of the usual starter), home minus away, fitted on final margins.
Writes qb_change.json (used by app/export_games.py) and prints the fit."""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from common import margin_ratings, market_ratings
from current_ratings import P

def usual_of(h, season):
    """the usual starter: most starts this season once the team has 2+ games, else most starts in its last 8 games
    (a new starter who took over in the offseason, e.g. Penix for Cousins, is the usual one by week 3)"""
    cur = [q for s_, q in h if s_ == season]
    pool = cur if len(cur) >= 2 else [q for _, q in h[-8:]]
    return pd.Series(pool).value_counts().index[0] if len(h) >= 3 else None

def starters(g):
    """per game: home/away flags 0 = usual starter, 1 = new backup (usual starter started last game), 2 = backup again"""
    hist = {}; fl = np.zeros((len(g), 2), int); usual = {}
    for i, r in enumerate(g.itertuples()):
        for j, (t, q) in enumerate(((r.home_team, r.home_qb_id), (r.away_team, r.away_qb_id))):
            h = hist.get(t, [])
            if len(h) >= 3 and isinstance(q, str):
                u = usual_of(h, r.season); usual[t] = u
                if q != u: fl[i, j] = 1 if h[-1][1] == u else 2
            if isinstance(q, str) and r.done: hist.setdefault(t, []).append((r.season, q))
    return fl, usual

M0, N0, DECAY = -0.05, 250.0, 0.7
def qb_games(pbp_dir, seasons):
    """per QB per game: dropbacks and EPA (cached for finished seasons in ../data/qb_games.parquet)"""
    cache = os.path.join(os.path.dirname(HERE), 'data', 'qb_games.parquet'); old = pd.read_parquet(cache) if os.path.exists(cache) else None
    have = set(old.season) if old is not None else set(); parts = [old] if old is not None else []
    for y in seasons:
        if y in have and y < max(seasons): continue
        f = f'{pbp_dir}/pbp_{y}.parquet'
        if not os.path.exists(f) and y == max(seasons):   # a fresh container: fetch this season's play-by-play
            import urllib.request; os.makedirs(pbp_dir, exist_ok=True)
            try: urllib.request.urlretrieve(f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet', f)
            except Exception as e: print('pbp download failed:', e)
        if not os.path.exists(f): continue
        p = pd.read_parquet(f, columns=['game_id', 'season', 'week', 'qb_dropback', 'id', 'qb_epa'])
        p = p[(p.qb_dropback == 1) & p.id.notna()]
        parts = [x[x.season != y] for x in parts] + [p.groupby(['game_id', 'season', 'week', 'id']).agg(db=('qb_epa', 'size'), epa=('qb_epa', 'sum')).reset_index()]
    q = pd.concat(parts, ignore_index=True)
    q[q.season < max(seasons)].to_parquet(cache)
    return q

def qb_value_fn(q):
    """value(qb, game_order) using only games before it. Returns dict game_id -> {qb: value} lookups via a running pass."""
    q = q.sort_values(['season', 'week', 'game_id'])
    run = {}; seen = {}; out = {}
    games = q.game_id.drop_duplicates().tolist(); by = dict(tuple(q.groupby('game_id')))
    last_season = None
    for gid in games:
        rows = by[gid]; s = int(rows.season.iloc[0])
        if s != last_season:
            run = {k: (v[0] * DECAY, v[1] * DECAY) for k, v in run.items()}; last_season = s
        out[gid] = {k: (v[0] + N0 * M0) / (v[1] + N0) for k, v in run.items()}   # snapshot before the game (only QBs seen so far)
        for r in rows.itertuples():
            e, d = run.get(r.id, (0.0, 0.0)); run[r.id] = (e + r.epa, d + r.db)
    final = {k: (v[0] + N0 * M0) / (v[1] + N0) for k, v in run.items()}
    return out, final

def depth_starters(n, pbp_dir):
    """2026-10-27 (J: PHI was favored with Hurts listed while Tanner McKee starts): games.csv's projected QB for upcoming
    games goes stale, so each team's next starter is its latest nflverse depth-chart QB1 (ESPN), as in the score rebuild
    (models/rebuild/football/live_nfl.py). If that QB is Out or Doubtful on the latest injury report (depth charts often
    keep an injured starter at QB1, e.g. Lamar Jackson in week 7), the starter of the team's last game stays instead."""
    import urllib.request, time
    season = int(n.season.max()); B = 'https://github.com/nflverse/nflverse-data/releases/download'
    fd, fi = f'{pbp_dir}/depth_charts_{season}.parquet', f'{pbp_dir}/inj_{season}.parquet'
    for u, f in ((f'{B}/depth_charts/depth_charts_{season}.parquet', fd), (f'{B}/injuries/injuries_{season}.parquet', fi)):
        if not os.path.exists(f) or os.path.getmtime(f) < time.time() - 12 * 3600:
            try: urllib.request.urlretrieve(u, f)
            except Exception as e: print('download failed', u, e)
    # finished games this season: where games.csv names a starter who threw no pass in play-by-play (it kept Hurts for PHI
    # week 7 although McKee threw every pass), the QB with the most attempts is used instead
    fp = f'{pbp_dir}/pbp_{season}.parquet'
    if os.path.exists(fp):
        pb = pd.read_parquet(fp, columns=['game_id', 'posteam', 'passer_player_id', 'passer_player_name']).dropna()
        cnt = pb.groupby(['game_id', 'posteam', 'passer_player_id']).size()
        threw = set(cnt.index)   # (game, team, qb) with 1+ attempt
        top = cnt.reset_index(name='c').sort_values('c').groupby(['game_id', 'posteam']).tail(1)
        top = {(r.game_id, r.posteam): r.passer_player_id for r in top.itertuples()}
        ids = pd.concat([n[['home_qb_id', 'home_qb_name']].set_axis(['i', 'nm'], axis=1), n[['away_qb_id', 'away_qb_name']].set_axis(['i', 'nm'], axis=1)]).dropna().drop_duplicates('i')
        nm = dict(zip(ids.i, ids.nm)); fixed = []
        for ix in n.index[n.done & n.season.eq(season)]:
            for side in ('home', 'away'):
                gid, tm, cur = n.at[ix, 'game_id'], n.at[ix, f'{side}_team'], n.at[ix, f'{side}_qb_id']
                q = top.get((gid, tm))
                # only where the listed starter threw no pass at all (a stale entry); otherwise games.csv stands
                if q and q != cur and (gid, tm, cur) not in threw:
                    fixed.append(f"{n.at[ix, 'game_id']} {n.at[ix, f'{side}_qb_name']} -> {nm.get(q, q)}"); n.at[ix, f'{side}_qb_id'] = q; n.at[ix, f'{side}_qb_name'] = nm.get(q, q)
        if fixed: print('finished-game starters fixed from play-by-play:', fixed)
    if not os.path.exists(fd): print('no depth chart; projected QBs from games.csv'); return n
    dc = pd.read_parquet(fd); dc['dt'] = pd.to_datetime(dc.dt)
    dc = dc[dc.pos_abb.eq('QB')].sort_values('dt'); dc = dc[dc.dt == dc.groupby('team').dt.transform('max')]
    out = set()
    if os.path.exists(fi):
        inj = pd.read_parquet(fi); inj = inj[inj.week == inj.week.max()]
        out = set(inj.gsis_id[inj.report_status.isin(['Out', 'Doubtful'])])
    done = n[n.done & n.season.eq(season)]
    last = {}
    for r in done.itertuples():
        last[r.home_team] = (r.home_qb_id, r.home_qb_name); last[r.away_team] = (r.away_qb_id, r.away_qb_name)
    pick = {}
    for t, x in dc.groupby('team'):
        x = x.sort_values('pos_rank'); q1 = x.iloc[0]
        pick[t] = (q1.gsis_id, q1.player_name) if q1.gsis_id not in out else last.get(t, (q1.gsis_id, q1.player_name))
    # manual overrides (J 2026-11-05: BAL traded for Brissett, ESPN depth chart lagged) from qb_overrides.json
    fo = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qb_overrides.json')
    if os.path.exists(fo):
        today = pd.Timestamp.now(tz='America/Denver').strftime('%Y-%m-%d')
        for t, o in json.load(open(fo)).get('teams', {}).items():
            if o.get('until', '9999') >= today: pick[t] = (o['id'], o['name']); print('QB override:', t, o['name'])
    up = ~n.done & n.season.eq(season)
    for side in ('home', 'away'):
        tm = n.loc[up, f'{side}_team']
        n.loc[up, f'{side}_qb_id'] = tm.map(lambda t: pick.get(t, (None, None))[0]).where(tm.isin(pick.keys()), n.loc[up, f'{side}_qb_id'])
        n.loc[up, f'{side}_qb_name'] = tm.map(lambda t: pick.get(t, (None, None))[1]).where(tm.isin(pick.keys()), n.loc[up, f'{side}_qb_name'])
    return n

def main(path, pbp_dir='/tmp/claude-0/nfldl'):
    n = pd.read_csv(path); n = n[n.season >= 2006].copy()
    n['done'] = n.result.notna(); n['date'] = pd.to_datetime(n.gameday)
    old_qb = {(r.game_id, s): getattr(r, f'{s}_qb_name') for r in n[~n.done & n.season.eq(n.season.max())].itertuples() for s in ('home', 'away')}
    n = depth_starters(n, pbp_dir)
    ch = {}
    for r in n[~n.done & n.season.eq(n.season.max())].itertuples():
        for s in ('home', 'away'):
            o, q = old_qb[(r.game_id, s)], getattr(r, f'{s}_qb_name')
            if isinstance(o, str) and o != q: ch[getattr(r, f'{s}_team')] = f'{o} -> {q}'
    print('starters changed from games.csv by the depth chart:', ch)
    n['date'] = pd.to_datetime(n.gameday); n = n.sort_values(['date', 'game_id']).reset_index(drop=True)
    n['home'], n['away'] = n.home_team, n.away_team; n['margin'] = n.home_score - n.away_score
    n['neutral'] = n.location.eq('Neutral'); n['line'] = n.spread_line; n['done'] = n.result.notna()
    fl, usual = starters(n)
    q = qb_games(pbp_dir, range(2016, int(n.season.max()) + 1)); snap, final = qb_value_fn(q)
    val = lambda gid, qb, done: (snap.get(gid, {}) if done else final).get(qb, M0)
    uq = {}; dq = np.zeros(len(n)); hist = {}
    for i, r in enumerate(n.itertuples()):
        d = []
        for t, qb in ((r.home_team, r.home_qb_id), (r.away_team, r.away_qb_id)):
            h = hist.get(t, []); u = usual_of(h, r.season)
            d.append(0.0 if (u is None or not isinstance(qb, str) or qb == u or r.season < 2017) else val(r.game_id, qb, r.done) - val(r.game_id, u, r.done))
            if isinstance(qb, str) and r.done: hist.setdefault(t, []).append((r.season, qb))
        dq[i] = d[0] - d[1]
    n['dq'] = dq
    p = P['NFL']; g = margin_ratings(n, neutral='neutral', **p['rt'])          # pre-game, future games get pre-game ratings too
    gl = market_ratings(g[g.line.notna()].copy(), 'line', neutral='neutral', out='mkt', **p['mkt'])
    g = g.join(gl[['mkt_pred']])
    W = json.load(open(os.path.join(os.path.dirname(HERE), 'blend_weights.json')))['NFL'] if os.path.exists(os.path.join(os.path.dirname(HERE), 'blend_weights.json')) else None
    m = g.done & g.season.between(2012, 2025) & g.mkt_pred.notna()
    X0 = np.column_stack([np.ones(m.sum()), g.rt_pred[m], g.mkt_pred[m]])
    new = (fl[:, 0] == 1).astype(float) - (fl[:, 1] == 1); cont = (fl[:, 0] == 2).astype(float) - (fl[:, 1] == 2)
    m = m & g.season.ge(2017); X0 = np.column_stack([np.ones(m.sum()), g.rt_pred[m], g.mkt_pred[m]])
    dqn = g.dq.values * (fl[:, 0] + fl[:, 1] > 0)   # value gap only where someone other than the usual starter starts
    newq = dqn * ((fl[:, 0] == 1) | (fl[:, 1] == 1)); contq = dqn - newq
    X = np.column_stack([X0, newq[m], contq[m]])
    out = {}
    for tgt in ('margin', 'line'):
        y = g[tgt][m].values; b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * r.var())
        b0, *_ = np.linalg.lstsq(X0, y, rcond=None); r0 = y - X0 @ b0
        out[tgt] = dict(new_per_epa=round(float(b[3]), 2), new_se=round(float(se[3]), 2), cont_per_epa=round(float(b[4]), 2), cont_se=round(float(se[4]), 2),
                        rmse=round(float(np.sqrt(np.mean(r ** 2))), 3), rmse_without=round(float(np.sqrt(np.mean(r0 ** 2))), 3))
        print(tgt, out[tgt])
    print('games with a backup start (either side):', int(((fl[m.values] > 0).any(1)).sum()), 'of', int(m.sum()))
    # out of sample: fit 2012-2019, test 2020-2025 margins
    tr = m & g.season.le(2019); te = m & g.season.ge(2020)
    def fit(mask, cols): A = np.column_stack(cols(mask)); return np.linalg.lstsq(A, g.margin[mask].values, rcond=None)[0]
    c1 = lambda k: [np.ones(k.sum()), g.rt_pred[k], g.mkt_pred[k]]; c2 = lambda k: c1(k) + [newq[k], contq[k]]
    e1 = g.margin[te].values - np.column_stack(c1(te)) @ fit(tr, c1); e2 = g.margin[te].values - np.column_stack(c2(te)) @ fit(tr, c2)
    bk = (fl[te.values] > 0).any(1)
    print(f'2020-25 held out RMSE: without {np.sqrt(np.mean(e1**2)):.3f}, with {np.sqrt(np.mean(e2**2)):.3f}; backup games only {np.sqrt(np.mean(e1[bk]**2)):.2f} -> {np.sqrt(np.mean(e2[bk]**2)):.2f} (n {bk.sum()})')
    up = g[~g.done & g.season.eq(g.season.max())]
    cur = {r.game_id: [int(fl[i, 0]), int(fl[i, 1]), r.home_qb_name, r.away_qb_name, round(float(newq[i] if fl[i].max() == 1 else contq[i]), 3)] for i, r in up.iterrows() if isinstance(r.home_qb_name, str)}
    qbname = dict(zip(n.home_qb_id, n.home_qb_name)) | dict(zip(n.away_qb_id, n.away_qb_name))
    J = dict(fit=out, upcoming=cur, usual={t: qbname.get(q, q) for t, q in usual.items()})
    json.dump(J, open(f'{HERE}/qb_change.json', 'w'), indent=1)
    for k, v in cur.items():
        if v[0] or v[1]: print('upcoming backup start:', k, v)
    return J

if __name__ == '__main__': main(*sys.argv[1:3])
