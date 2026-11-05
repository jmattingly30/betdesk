"""Our player prop models vs real 2026 sportsbook prop prices (OddsPapi history, op/us/hist_<lg>.jsonl from pull.py).
J's rule: grade OUR unblended model. For every (game, player, stat, line, book) with both sides priced:
  market  = that book's no-vig over chance at the open and at the close (last price before start)
  model   = our calibrated P(over) at that exact line
Reports log loss / Brier (model vs open vs close), bets at the open when our EV clears a threshold (win %, ROI at the
open price, CLV = close no-vig chance x open price - 1), and the same at the close.

  python3 props_backtest.py nba        # 2026 regular season + playoffs, out-of-sample walk-forward predictions
  python3 props_backtest.py nhl        # 2026 part of 2025-26, fit 2023-25 (as nhl-player/model.py's test)
  python3 props_backtest.py live       # NBA/NHL/NFL projections exactly as Bet Desk showed them (grading/history)
Player ids: OddsPapi keys props by its own player id with no name in history, so ids are matched to our players per
game (Hungarian assignment on line vs our projection over every stat priced), then fixed by a vote across games and
by exact names from upcoming odds (op/us/player_ids.json, pull.py names). Writes results/<mode>_*.csv + a log."""
import os, sys, json, gzip, re, unicodedata, collections, datetime as dt
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import stats as st
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__)); SA = os.path.normpath(f'{HERE}/../..'); US = f'{SA}/op/us'
RES = f'{HERE}/results'; os.makedirs(RES, exist_ok=True)
ET = ZoneInfo('America/New_York')

STAT = {11: {'Points': 'pts', 'Rebounds': 'reb', 'Assists': 'ast', '3 Point FG': 'fg3m', 'Steals': 'stl', 'Blocks': 'blk',
             'Turnovers': 'tov', 'Points + Assists + Rebounds': 'pra'},
        15: {'Shots On Goal': 'sog', 'Points': 'points', 'Goals': 'goals', 'Assists': 'assists', 'Saves': 'saves', 'Blocks': 'blk'},
        14: {'Pass Yards': 'pass_yds', 'Receiving Yards': 'rec_yds', 'Rush Yards': 'rush_yds', 'Pass Attempts': 'pass_att',
             'Pass Completions': 'cmp', 'Rush Attempts': 'carries', 'Receptions': 'rec', 'TD Passes': 'pass_td'}}
SPORT = {'nba': 11, 'nhl': 15, 'nfl': 14}
TEAMS = {
 'nba': dict(zip(['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers', 'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets', 'Indiana Pacers', 'LA Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks', 'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs', 'Toronto Raptors', 'Utah Jazz', 'Washington Wizards'],
                 ['ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GS', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NO', 'NY', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SA', 'TOR', 'UTAH', 'WSH'])),
 'nhl': dict(zip(['Anaheim Ducks', 'Boston Bruins', 'Buffalo Sabres', 'Calgary Flames', 'Carolina Hurricanes', 'Chicago Blackhawks', 'Colorado Avalanche', 'Columbus Blue Jackets', 'Dallas Stars', 'Detroit Red Wings', 'Edmonton Oilers', 'Florida Panthers', 'Los Angeles Kings', 'Minnesota Wild', 'Montreal Canadiens', 'Nashville Predators', 'New Jersey Devils', 'New York Islanders', 'New York Rangers', 'Ottawa Senators', 'Philadelphia Flyers', 'Pittsburgh Penguins', 'San Jose Sharks', 'Seattle Kraken', 'St. Louis Blues', 'Tampa Bay Lightning', 'Toronto Maple Leafs', 'Utah Mammoth', 'Vancouver Canucks', 'Vegas Golden Knights', 'Washington Capitals', 'Winnipeg Jets'],
                 ['ANA', 'BOS', 'BUF', 'CGY', 'CAR', 'CHI', 'COL', 'CBJ', 'DAL', 'DET', 'EDM', 'FLA', 'LAK', 'MIN', 'MTL', 'NSH', 'NJD', 'NYI', 'NYR', 'OTT', 'PHI', 'PIT', 'SJS', 'SEA', 'STL', 'TBL', 'TOR', 'UTA', 'VAN', 'VGK', 'WSH', 'WPG'])),
 'nfl': dict(zip(['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs', 'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants', 'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers', 'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders'],
                 ['ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LA', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS']))}


def norm(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', re.sub(r'[^a-z ]', '', s)).split())


def short(s):  # "Brent Burns" / "Burns, Brent" -> "b burns" (NHL feed uses "B. Burns")
    s = str(s)
    if ',' in s: last, first = [x.strip() for x in s.split(',', 1)]; s = f'{first} {last}'
    n = norm(s).split()
    return f'{n[0][0]} {" ".join(n[1:])}' if len(n) > 1 else ' '.join(n)


def load_props(lg):
    """Long table of two-way props: one row per fixture/book/stat/line/player with open+close prices of both sides."""
    sp = SPORT[lg]
    M = {str(m['marketId']): m for m in json.load(open(f'{SA}/op/markets.json')) if m['sportId'] == sp and m['playerProp']}
    want = {}
    for mid, m in M.items():
        nm = m['marketName']
        if not nm.startswith('Over Under Player ') or len(m['outcomes']) != 2: continue
        base = nm[len('Over Under Player '):].replace(' (incl. overtime)', '')
        if base in STAT[sp]:
            want[mid] = (STAT[sp][base], m['handicap'], {str(o['outcomeId']): o['outcomeName'] for o in m['outcomes']})
    rows, fx = [], []
    path = f'{US}/hist_{lg}.jsonl'
    for line in open(path):
        try: r = json.loads(line)
        except ValueError: continue   # a line still being written by pull.py
        if 'err' in r: continue
        d = dt.datetime.fromisoformat(r['start'].replace('Z', '+00:00')).astimezone(ET).date().isoformat()
        T = TEAMS[lg]
        fx.append(dict(fid=r['fid'], date=d, start=r['start'], home=T.get(r['home']), away=T.get(r['away'])))
        for bk, bm in r['books'].items():
            for mid, outs in bm.items():
                if mid not in want: continue
                stat, L, on = want[mid]
                side = {on[o]: v for o, v in outs.items() if o in on}
                if 'Over' not in side or 'Under' not in side: continue
                for pid in set(side['Over']) & set(side['Under']):
                    o, u = side['Over'][pid], side['Under'][pid]
                    alive = min(o[4], u[4]) if len(o) > 4 and len(u) > 4 else None
                    rows.append((r['fid'], bk, stat, L, pid, o[0][1], u[0][1], o[2][1], u[2][1], max(o[0][0], u[0][0]), max(o[2][0], u[2][0]),
                                 (o[1] or [None, None])[1], (u[1] or [None, None])[1], o[2][2], alive))
    P = pd.DataFrame(rows, columns=['fid', 'book', 'stat', 'line', 'opid', 'o_open', 'u_open', 'o_close', 'u_close', 'open_secs', 'close_secs',
                                    'o_d24', 'u_d24', 'limit', 'alive'])
    for k in ('open', 'close', 'd24'):
        io, iu = 1 / P[f'o_{k}'].astype(float), 1 / P[f'u_{k}'].astype(float)
        P[f'p_{k}'] = io / (io + iu); P[f'hold_{k}'] = io + iu - 1
    # Each handicap is its own market, so a line the book moved off keeps its last price from hours ago. The OPENING
    # lines are those first posted within 30 min of the earliest; the CLOSING lines are those still up at the start
    # (alive), or (older rows) last priced within 30 min of the latest.
    g = P.groupby(['fid', 'book', 'stat', 'opid'])
    # junk ladders: some feeds (seen on FanDuel NBA points) carry 5-30 lines per player all priced near even, which
    # cannot all be real; drop any player-stat with 3+ lines whose over chance does not fall as the line rises
    # (a real ladder's over chance falls as the line rises; flag groups where it doesn't)
    nl = g.line.transform('nunique')
    rho = g[['line', 'p_close']].apply(lambda x: x.line.corr(x.p_close, method='spearman') if x.line.nunique() >= 3 else -1.0)
    bad = set(rho[rho > -0.5].index)
    P = P[[k not in bad for k in zip(P.fid, P.book, P.stat, P.opid)]].copy()
    # cross-book check (no use of our model): a book whose closing main line for a player-stat sits far from the other
    # books' is mislabelled (FanDuel NBA 'points' carried another market for ~9% of players); drop that book's rows
    g = P.groupby(['fid', 'book', 'stat', 'opid'])
    alive_main = P[P.alive.fillna(1) == 1].assign(d=lambda x: (x.p_close - 0.5).abs()).sort_values('d').drop_duplicates(['fid', 'book', 'stat', 'opid'])
    med = alive_main.groupby(['fid', 'stat', 'opid']).line.transform('median'); nb = alive_main.groupby(['fid', 'stat', 'opid']).book.transform('nunique')
    off = alive_main[(nb >= 2) & ((alive_main.line - med).abs() > np.maximum(2.5, 0.15 * med.abs()))]
    badb = set(zip(off.fid, off.book, off.stat, off.opid))
    P = P[[k not in badb for k in zip(P.fid, P.book, P.stat, P.opid)]].copy()
    g = P.groupby(['fid', 'book', 'stat', 'opid'])
    P['is_open'] = P.open_secs >= g.open_secs.transform('max') - 1800
    recent = P.close_secs <= g.close_secs.transform('min') + 1800
    P['is_close'] = np.where(P.alive.notna(), (P.alive == 1) & (P.close_secs <= g.close_secs.transform('min') + 3 * 3600), recent)
    return P, pd.DataFrame(fx)


def name_ids(P, cand, out_json, prob=False):
    """pid -> our player key. cand: DataFrame fid, key, stat, mu (our projection) for every player who could be in that game."""
    exact = {}
    pj = f'{US}/player_ids.json'
    if os.path.exists(pj): exact = json.load(open(pj))
    main = (P.assign(dist=(P.p_close - 0.5).abs()).sort_values('dist').drop_duplicates(['fid', 'opid', 'stat']))
    votes = collections.defaultdict(collections.Counter)
    for fid, g in main.groupby('fid'):
        c = cand[cand.fid == fid]
        if c.empty: continue
        pids = sorted(g.opid.unique()); keys = sorted(c.key.unique())
        L = g.pivot_table(index='opid', columns='stat', values='line', aggfunc='first').reindex(pids)
        Mu = c.pivot_table(index='key', columns='stat', values='mu', aggfunc='first').reindex(keys)
        cols = [s for s in L.columns if s in Mu.columns]
        if not cols: continue
        Lv, Mv = L[cols].to_numpy(float), Mu[cols].to_numpy(float)
        if prob:   # low-count props (NHL: nearly every line is 0.5) -> compare the book's chance with a Poisson on our mean
            Pv = g.pivot_table(index='opid', columns='stat', values='p_close', aggfunc='first').reindex(pids)[cols].to_numpy(float)
            diff = np.abs(Pv[:, None, :] - st.poisson.sf(np.floor(Lv[:, None, :]), np.clip(Mv[None, :, :], 1e-3, None))) / 0.08
            diff = np.where(np.isnan(Mv[None, :, :]) | np.isnan(Lv[:, None, :]), np.nan, diff)
        else:
            diff = np.abs(Lv[:, None, :] - Mv[None, :, :]) / np.sqrt(np.nan_to_num(Mv[None, :, :], nan=1) + 1)
        cnt = (~np.isnan(diff)).sum(2)
        with np.errstate(all='ignore'), __import__('warnings').catch_warnings():
            __import__('warnings').simplefilter('ignore'); cost = np.where(cnt > 0, np.nanmean(diff, 2), 9.0)
        cost = np.nan_to_num(cost, nan=9.0) + 0.3 / np.maximum(cnt, 1)  # more stats agreeing = more trust
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            if cost[i, j] < 1.2: votes[pids[i]][keys[j]] += 1
    ids = {}
    for pid, v in votes.items():
        k, n = v.most_common(1)[0]
        if n >= 2 and n / sum(v.values()) >= 0.6 or (n == 1 and sum(v.values()) == 1): ids[pid] = k
    for pid, nm in exact.items(): ids[pid] = nm  # exact names win (normalised by caller)
    json.dump({k: v for k, v in ids.items()}, open(out_json, 'w'))
    return ids, votes


ll = lambda p, y: -np.mean(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1 - 1e-4)))


def evaluate(J, tag, log):
    """J: one row per fixture/book/stat/line/player with p_model, is_open/is_close, prices, actual."""
    J = J[J.actual != J.line].copy(); J['over'] = (J.actual > J.line).astype(int)
    say = lambda *a: (print(*a), log.write(' '.join(map(str, a)) + '\n'))
    # fair close chance at every line (for CLV): same line if still up at the close, else the closing main line's
    # no-vig chance shifted by how much our model's chance changes between the two lines
    C = J[J.is_close].assign(d=lambda x: (x.p_close - 0.5).abs()).sort_values('d').drop_duplicates(['fid', 'book', 'stat', 'opid'])
    C = C.set_index(['fid', 'book', 'stat', 'opid'])[['line', 'p_close', 'p_model']].rename(columns=lambda c: 'cm_' + c)
    J = J.join(C, on=['fid', 'book', 'stat', 'opid'])
    same = J[J.is_close].groupby(['fid', 'book', 'stat', 'opid', 'line']).p_close.first()
    J['fair_close'] = same.reindex(pd.MultiIndex.from_frame(J[['fid', 'book', 'stat', 'opid', 'line']])).to_numpy()
    J['fair_close'] = J.fair_close.fillna((J.cm_p_close + J.p_model - J.cm_p_model).clip(0.01, 0.99))
    acc = []
    for when, flag in (('open', 'is_open'), ('close', 'is_close')):
        X = J[J[flag]]
        for (bk, s), g in list(X.groupby(['book', 'stat'])) + [(('all', 'all'), X)]:
            if len(g) < 30: continue
            pm = g[f'p_{when}']
            acc.append(dict(when=when, book=bk, stat=s, n=len(g), players=g[['fid', 'opid']].drop_duplicates().shape[0], over_rate=g.over.mean(),
                            ll_model=ll(g.p_model, g.over), ll_book=ll(pm, g.over), ll_blend=ll(0.5 * g.p_model + 0.5 * pm, g.over),
                            brier_model=((g.p_model - g.over) ** 2).mean(), brier_book=((pm - g.over) ** 2).mean(),
                            corr=np.corrcoef(g.p_model, pm)[0, 1], avg_hold=g[f'hold_{when}'].mean()))
    # size check: our mean vs the closing main line vs what happened (props void if the player sits, so all three are
    # conditional on playing)
    mu = J['mu_' + J.stat.iloc[0]] if False else pd.Series([r.get('mu_' + st_) for st_, r in zip(J.stat, J.to_dict('records'))], index=J.index)
    K = J.assign(mu=mu)[J.is_close].assign(d=lambda x: (x.p_close - 0.5).abs()).sort_values('d').drop_duplicates(['fid', 'opid', 'stat'])
    S = K.groupby('stat').agg(n=('line', 'size'), our_mean=('mu', 'mean'), close_line=('line', 'mean'), actual=('actual', 'mean'),
                              our_mae=('mu', lambda x: np.nan)).drop(columns='our_mae')
    S['our_mae'] = K.groupby('stat').apply(lambda g: (g.actual - g.mu).abs().mean()); S['line_mae'] = K.groupby('stat').apply(lambda g: (g.actual - g.line).abs().mean())
    say(f'\n== {tag}: size check at the closing main line (one per player-stat-game)\n', S.round(2).to_string())
    S.to_csv(f'{RES}/{tag}_size.csv')
    A = pd.DataFrame(acc); say(f'\n== {tag}: accuracy at the opening lines and at the closing lines (log loss: lower is better; book = no-vig)\n', A.round(4).to_string(index=False))
    bets = []
    for when, flag in (('open', 'is_open'), ('close', 'is_close')):
        X = J[J[flag]]
        for books in (('pinnacle',), ('draftkings', 'fanduel'), ('pinnacle', 'draftkings', 'fanduel')):
            Y = X[X.book.isin(books)]
            for thr in (0.0, 0.03, 0.05, 0.08, 0.12):
                for side in ('Over', 'Under'):
                    pr = Y[f'{side[0].lower()}_{when}'].astype(float)
                    pm = Y.p_model if side == 'Over' else 1 - Y.p_model
                    ev = pm * pr - 1
                    e = Y[ev > thr].assign(ev=ev[ev > thr], price=pr[ev > thr])
                    e = e.sort_values('ev', ascending=False).drop_duplicates(['fid', 'opid', 'stat'])  # one bet per player-stat-game
                    if not len(e): continue
                    win = (e.over == 1) if side == 'Over' else (e.over == 0)
                    pnl = np.where(win, e.price - 1, -1.0)
                    fc = e.fair_close if side == 'Over' else 1 - e.fair_close
                    bets.append(dict(when=when, books='+'.join(b[:3] for b in books), ev_over=thr, side=side, bets=len(e), win=win.mean(), roi=pnl.mean(),
                                     roi_se=pnl.std() / np.sqrt(len(e)), clv=(fc * e.price - 1).mean() if when == 'open' else np.nan, avg_price=e.price.mean()))
    B = pd.DataFrame(bets); say(f'\n== {tag}: bets when our EV beats the price (one per player-stat-game, best of those books; CLV = fair close x price - 1)\n', B.round(3).to_string(index=False))
    A.to_csv(f'{RES}/{tag}_accuracy.csv', index=False); B.to_csv(f'{RES}/{tag}_bets.csv', index=False)
    return A, B


def match_games(F, games, lg):
    """fixtures -> our game ids by US Eastern date + the two teams. games: DataFrame d (YYYY-MM-DD), game_id, team."""
    gm = games.groupby(['d', 'game_id']).team.apply(frozenset).reset_index()
    F = F.assign(pair=[frozenset([h, a]) for h, a in zip(F.home, F.away)])
    return F.merge(gm, left_on=['date', 'pair'], right_on=['d', 'team'], how='inner')[['fid', 'game_id']]


def run(lg, T, stats, p_over, keyf=norm, hist=None, Fm=None, prob=False):
    """T: our out-of-sample predictions, one row per player-game: game_id, d, team, key, mu_<stat>, <stat> (actual).
    p_over(J) -> our calibrated P(over) for each joined row (J has stat, line and T's columns)."""
    log = open(f'{RES}/{lg}_log.txt', 'w')
    P, F = load_props(hist or lg)
    say = lambda *a: (print(*a, flush=True), log.write(' '.join(map(str, a)) + '\n'))
    say(lg, 'prop rows', len(P), 'fixtures', len(F), 'fixtures with props by book', P.groupby('book').fid.nunique().to_dict(),
        'stats', P.groupby('stat').size().to_dict())
    F = match_games(F, T, lg) if Fm is None else Fm; T = T.merge(F, on='game_id')
    T = T[~T.duplicated(['fid', 'key'], keep=False)]   # two players with the same name key in one game: can't tell apart
    say('fixtures matched to our games', F.fid.nunique())
    cand = T.melt(id_vars=['fid', 'key'], value_vars=[f'mu_{s}' for s in stats if f'mu_{s}' in T], var_name='stat', value_name='mu')
    cand['stat'] = cand.stat.str[3:]
    ids, votes = name_ids(P, cand, f'{RES}/{lg}_player_ids.json', prob=prob)
    P['key'] = P.opid.map(ids).map(lambda x: keyf(x) if isinstance(x, str) else None)
    say('player ids named', len(ids), 'of', P.opid.nunique(), '| prop rows named', round(P.key.notna().mean(), 3))
    J = P.dropna(subset=['key']).merge(T, on=['fid', 'key'], how='inner')
    J = J[J.stat.isin(stats)].copy()
    J['p_model'] = p_over(J)
    J['actual'] = [r[s] for s, r in zip(J.stat, J.to_dict('records'))]
    try: J.to_parquet(f'{RES}/{lg}_joined.parquet')
    except Exception: J.astype({c: str for c in J.columns if J[c].dtype == object}).to_parquet(f'{RES}/{lg}_joined.parquet')
    say('joined rows', len(J), J.groupby('book').size().to_dict())
    evaluate(J, lg, log)


def run_fixed(tag, lg, T, Fm, stats, p_over):
    run(tag, T, stats, p_over, hist=lg, Fm=Fm)


def nba():
    sys.path.insert(0, f'{SA}/models/nba-player'); import model as nm, pickle
    tp = pd.read_parquet(f'{SA}/models/nba-player/data/test_predictions.parquet')
    tp = tp[tp.date >= '2026-01-01'].copy(); tp['d'] = tp.date.dt.strftime('%Y-%m-%d'); tp['key'] = tp.name.map(norm)
    pk = f'{RES}/nba_fit_2019_2024.pkl'  # dispersion + temperature from the fit that made test_predictions (never saw 2024-25 on)
    if os.path.exists(pk): m = pickle.load(open(pk, 'rb'))
    else:
        f = pd.read_parquet(f'{SA}/models/nba-player/data/features.parquet'); f = f[nm.usable(f)]
        m = nm.fit(f[f.season.between(2019, 2024)]); pickle.dump(m, open(pk, 'wb'))
    run('nba', tp, ['pts', 'reb', 'ast', 'fg3m', 'stl', 'blk', 'tov', 'pra'],
        lambda J: [m.prob_over(r, r.stat, r.line) for _, r in J.iterrows()])


def nhl():
    """2025-26 out of sample exactly as models/nhl-player/model.py: fit 2023-24 -> calibrate on 2024-25; fit 2023-25 -> test 2025-26."""
    sys.path.insert(0, f'{SA}/models/nhl-player'); import model as hm
    d = pd.read_parquet(f'{SA}/models/nhl-player/data/player_games.parquet'); sk, gl = hm.frames(d)
    m1 = hm.fit_means(sk[sk.season == 2023], gl[gl.season == 2023])
    cal = hm.calibrate(*hm.predict(m1, sk[sk.season == 2024].copy(), gl[gl.season == 2024].copy()))
    m2 = hm.fit_means(sk[sk.season <= 2024], gl[gl.season <= 2024])
    t_sk, t_gl = hm.predict(m2, sk[sk.season == 2025].copy(), gl[gl.season == 2025].copy())
    T = pd.concat([t_sk, t_gl]); T = T[T.game_date >= '2026-01-01'].copy()
    T['d'] = T.game_date.dt.strftime('%Y-%m-%d'); T['key'] = T.player_name.map(short)
    for s in hm.SK_STATS + ['saves']: T[f'mu_{s}'] = T[f'pred_{s}']
    def p_over(J):
        out = np.full(len(J), np.nan)
        for s, idx in J.groupby('stat').indices.items():
            g = J.iloc[idx]; c = cal[s]
            out[idx] = hm.temp(hm.nb_p_over(g[f'mu_{s}'].to_numpy(), c['alpha'], g.line.to_numpy()), c['temp'])
        return out
    run('nhl', T, ['sog', 'points', 'goals', 'assists', 'blk', 'saves'], p_over, keyf=short, prob=True)


YARDS = {'pass_yds', 'rush_yds', 'rec_yds', 'rush_rec_yds'}


def desk_p_over(m, L, po, lines, stat, sd_ratio):
    """Bet Desk stores mean m and over chance po at its fair line L; rebuild a distribution matching both and read
    P(over) at the book lines. Counts: negative binomial (alpha solved); yards: normal (sd solved, else sd_ratio x m)."""
    lines = np.asarray(lines, float)
    if stat in YARDS:
        z = st.norm.ppf(1 - po) if 0.02 < po < 0.98 else 0
        sd = (L - m) / z if abs(z) > 0.05 and (L - m) / z > 0 else sd_ratio * max(m, 1)
        return 1 - st.norm.cdf((lines - m) / sd)
    if m <= 0: return np.zeros(len(lines))
    best, ba = 9, 0.0
    for a in (0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5):
        p = st.poisson.sf(np.floor(L), m) if a == 0 else st.nbinom.sf(np.floor(L), 1 / a, (1 / a) / (1 / a + m))
        if abs(p - po) < best: best, ba = abs(p - po), a
    return st.poisson.sf(np.floor(lines), m) if ba == 0 else st.nbinom.sf(np.floor(lines), 1 / ba, (1 / ba) / (1 / ba + m))


def live():
    """Bet Desk's projections exactly as shown before each game (models/grading/history/<SPORT>.json.gz, snapshot
    from 2026-10-15), graded against the books. Projections are refreshed up to game day, so the CLOSE is the fair
    comparison; the open comparison flatters us (we saw later news than the opener)."""
    sys.path.insert(0, f'{SA}/models/grading'); import grade
    only = sys.argv[2].split(',') if len(sys.argv) > 2 else ['nba', 'nhl', 'nfl']
    for lg, SP in (('nba', 'NBA'), ('nhl', 'NHL'), ('nfl', 'NFL')):
        if lg not in only: continue
        if not os.path.exists(f'{US}/hist_{lg}.jsonl'): continue
        H = pd.DataFrame(json.load(gzip.open(f'{SA}/models/grading/history/{SP}.json.gz')).values())
        R = grade.results(SP, grade.fetch(SP))
        A = pd.DataFrame([dict(g=k[0], t=k[1], n=k[2], **v) for k, v in R.items()])
        if A.empty: continue
        H = H.merge(A.drop(columns=[c for c in A.columns if c.startswith('_')]), on=['g', 't', 'n'], how='inner')
        H['key'] = H.n.map(short if lg == 'nhl' else norm)
        if lg == 'nba': H['g'] = [f"{g}|{'-'.join(sorted([t, o]))}" for g, t, o in zip(H.g, H.t, H.o)]  # desk keys NBA games by date only
        # one row per player-game with mu_<stat> + actual <stat>
        W = H.pivot_table(index=['g', 't', 'o', 'key'], columns='stat', values='m', aggfunc='first')
        W.columns = [f'mu_{c}' for c in W.columns]; W = W.reset_index()
        for s in H.stat.unique():
            W = W.merge(H[H.stat == s][['g', 't', 'key', s]].drop_duplicates(['g', 't', 'key']), on=['g', 't', 'key'], how='left') if s in H else W
        D = H.set_index(['g', 't', 'key', 'stat'])[['m', 'L', 'po']]
        if lg == 'nfl':
            T = W.copy()
            T['home'] = [g.split('_')[3] for g in W.g]; T['away'] = [g.split('_')[2] for g in W.g]
            P, F = load_props(lg)
            F = F[F.date.between('2026-10-14', '2026-11-03')]   # snapshot weeks 6-8 only (pairs repeat across a season)
            F = F.assign(pair=[frozenset([h, a]) for h, a in zip(F.home, F.away)])
            T['pair'] = [frozenset([h, a]) for h, a in zip(T.home, T.away)]
            gid = T.drop_duplicates('g').set_index('pair').g
            F['game_id'] = F.pair.map(gid)   # one 2026 regular-season game per pair in a 3-week window
            T = T.rename(columns={'g': 'game_id'}); T['team'] = T.t
            Fm = F.dropna(subset=['game_id'])[['fid', 'game_id']]
        else:
            T = W.rename(columns={'g': 'game_id'}); T['team'] = T.t
            gd = H.drop_duplicates('g').set_index('g').d
            T['d'] = T.game_id.map(gd)
        sd = {s: 0.6 for s in YARDS}
        def p_over(J, lg=lg):
            out = np.full(len(J), np.nan)
            for (g, t, k, s), idx in J.groupby(['game_id', 'team', 'key', 'stat']).indices.items():
                try: m, L, po = D.loc[(g, t, k, s)]
                except KeyError: continue
                out[idx] = desk_p_over(m, L, po, J.line.to_numpy()[idx], s, 0.6)
            return out
        stats = sorted(set(H.stat) & set(STAT[SPORT[lg]].values()))
        if lg == 'nfl': run_fixed('live_' + lg, lg, T, Fm, stats, p_over)
        else: run('live_' + lg, T, stats, p_over, keyf=short if lg == 'nhl' else norm, hist=lg, prob=lg == 'nhl')


if __name__ == '__main__':
    {'nba': nba, 'nhl': nhl, 'live': live}[sys.argv[1]]()
