"""APV (J's adjusted EPA) inputs for the NFL player model: roadmap steps 5 and 6 (J 2026-11-16).

Step 5 (rushing yards):
  opp_rapv  what the opponent's defense has allowed on running back runs, in APV per carry, adjusted for the offenses it
            faced with ratings from earlier weeks only (this season + half of last season, shrunk 60 carries toward 0).
            In the rush_yds efficiency GLM. Backtest 2024-25 (RBs, actual carries): yards error 15.086 -> 15.072.
  apv_r     the runner's own to-date rush APV per carry (this + last season, shrunk 146 carries). Kept as a column but NOT
            in the model: once his yards per carry is in, it added nothing (15.072 -> 15.069, negative coefficient).
Step 6 (rookie QB passing yards):
  rook_qb   1 in a QB's draft season; rook_apv = his college APV+ minus 115, per 10 points (0 if unknown or not a rookie).
            Efficiency-only test 2022-25 rookie QB starts (n 223, actual attempts): 39.52 -> 39.00 (flag alone 39.25).
            But in the full backtest (predicted attempts, 2024-25) rookie QB passing-yard error went 62.08 -> 62.66, worse
            in both seasons (bias +3.7 -> -3.1), so it is NOT in the live model. Columns kept for a later shadow test.

python3 apv_features.py <dl_dir> <model_dir>   -> <model_dir>/data/apv_games.parquet (rebuilt from pbp_YYYY.parquet; ~10 s)
add(df, model_dir) adds the four columns to a features frame (zeros if apv_games.parquet is missing).

Play value is the same as the APV stat (apv_nfl.py): nflverse EPA with fumble/INT luck set to the zone average,
offense/defense ridge ratings by play type (run / dropback) from earlier weeks only, prior = half of last season's final.
"""
import sys, os, glob, numpy as np, pandas as pd

SLOPE_RUSH = 1.35
LAM = 150.0
K_RUN, K_OPP = 146.0, 60.0
ROOK_CENTER = 115.0
C = ['game_id', 'season', 'week', 'posteam', 'defteam', 'home_team', 'play_type', 'epa', 'yardline_100', 'fumble', 'interception',
     'qb_kneel', 'qb_spike', 'rusher_player_id', 'qb_scramble']
HERE = os.path.dirname(os.path.abspath(__file__))


def ridge(tg, prior):
    teams = sorted(set(tg.team) | set(tg.opp) | set(prior[0].keys())); T = len(teams); ix = {t: i for i, t in enumerate(teams)}; p = 2 + 2 * T
    A = np.zeros((p, p)); b = np.zeros(p); P0 = np.zeros(p); P0[0] = prior[2]
    for t, i in ix.items(): P0[2 + i] = prior[0].get(t, 0.0); P0[2 + T + i] = prior[1].get(t, 0.0)
    if len(tg):
        k = np.arange(len(tg)); X = np.zeros((len(tg), p)); X[:, 0] = 1; X[:, 1] = np.where(tg.home, .5, -.5)
        X[k, 2 + np.array([ix[t] for t in tg.team])] = 1; X[k, 2 + T + np.array([ix[t] for t in tg.opp])] = 1
        w = tg.n.values.astype(float); A = X.T @ (X * w[:, None]); b = X.T @ (tg.y.values * w)
    pen = np.r_[1e-2, 30.0, np.full(2 * T, LAM)]; A[np.diag_indices(p)] += pen
    c = np.linalg.solve(A, b + pen * P0)
    return c[0], c[1], {t: c[2 + i] for t, i in ix.items()}, {t: c[2 + T + i] for t, i in ix.items()}


def build_games(dl, pos):
    """Per player-game rush credit (n_rush, v_rush) and per defense-game RB-run APV allowed (d_n, d_adj)."""
    prev = None; P_, D_ = [], []
    for f in sorted(glob.glob(f'{dl}/pbp_*.parquet')):
        y = int(f[-12:-8])
        if y < 2017:
            continue
        d = pd.read_parquet(f, columns=C)
        s = d[d.posteam.notna() & d.play_type.isin(['run', 'pass']) & d.epa.notna() & (d.qb_kneel != 1) & (d.qb_spike != 1)].copy()
        if s.empty:
            continue
        s['wk'] = s.week.astype(int); s['scr'] = s.qb_scramble == 1
        s['kind'] = np.where((s.play_type == 'pass') | s.scr, 'P', 'R'); s['team'] = s.posteam; s['opp'] = s.defteam; s['is_home'] = s.posteam == s.home_team
        s['z'] = np.digitize(s.yardline_100, [20, 50, 80]); s['fum'] = s.fumble == 1; s['intc'] = s.interception == 1
        fm = s[s.fum].groupby('z').epa.mean(); im = s[s.intc].groupby('z').epa.mean()
        s['X4'] = np.where(s.fum, s.z.map(fm), np.where(s.intc, s.z.map(im), s.epa))
        TG = s.groupby(['kind', 'wk', 'game_id', 'team', 'opp']).agg(y=('X4', 'mean'), n=('X4', 'size'), home=('is_home', 'first')).reset_index()
        e_def = np.zeros(len(s)); e_off = np.zeros(len(s)); mu_a = np.zeros(len(s)); final = {}
        for kind in ('R', 'P'):
            pr = ({t: .5 * v for t, v in prev[kind][2].items()}, {t: .5 * v for t, v in prev[kind][3].items()}, prev[kind][0]) if prev else \
                 ({}, {}, TG[TG.kind == kind].y.mean())
            for w in sorted(s.wk.unique()):
                mu, hm, o_, d_ = ridge(TG[(TG.kind == kind) & (TG.wk < w)], pr)
                m = ((s.kind == kind) & (s.wk == w)).values
                h = hm * np.where(s.is_home.values[m], .5, -.5)
                e_def[m] = mu + h + np.array([d_.get(o, 0.0) for o in s.opp.values[m]])     # expected given the defense
                e_off[m] = mu + h + np.array([o_.get(t, 0.0) for t in s.team.values[m]])    # expected given the offense
                mu_a[m] = mu
            final[kind] = ridge(TG[TG.kind == kind], pr)
        prev = final
        m = ((s.kind == 'R') & s.rusher_player_id.notna()).values
        ru = s[m].copy()
        ru['c'] = ru.X4 - (mu_a[m] + SLOPE_RUSH * (e_def[m] - mu_a[m]))     # runner credit vs a league-average defense
        ru['adj'] = ru.X4 - e_off[m]                                         # what the defense allowed vs this offense's norm
        ru['pos'] = ru.rusher_player_id.map(pos)
        P_.append(ru.groupby(['game_id', 'season', 'wk', 'rusher_player_id']).agg(n_rush=('c', 'size'), v_rush=('c', 'sum')).reset_index()
                  .rename(columns={'rusher_player_id': 'pid'}))
        rb = ru[ru.pos == 'RB']
        D_.append(rb.groupby(['game_id', 'season', 'wk', 'opp']).agg(d_n=('adj', 'size'), d_adj=('adj', 'sum')).reset_index())
        print('apv', y, flush=True)
    P = pd.concat(P_).assign(kind='p'); D = pd.concat(D_).assign(kind='d').rename(columns={'opp': 'key'})
    P = P.rename(columns={'pid': 'key'})
    return pd.concat([P, D], ignore_index=True).rename(columns={'wk': 'week'})


def _to_date(rows, games, key, ncol, vcol, prev_w):
    """For each row (key, season, week): sums over `games` this season before `week` + prev_w x last season."""
    g = games[[key, 'season', 'week', ncol, vcol]].groupby([key, 'season', 'week'], as_index=False).sum()
    r = rows[[key, 'season', 'week']].drop_duplicates().assign(**{ncol: 0.0, vcol: 0.0, '_r': 1})
    a = pd.concat([g.assign(_r=0), r], ignore_index=True).sort_values([key, 'season', 'week', '_r'], kind='mergesort')
    # a row for week w sorts after any game in week w, so exclude games of the same week: use games with week < w
    a['cn'] = a.groupby([key, 'season'])[ncol].cumsum(); a['cv'] = a.groupby([key, 'season'])[vcol].cumsum()
    same = a[a._r == 0].groupby([key, 'season', 'week'])[[ncol, vcol]].sum()
    out = a[a._r == 1][[key, 'season', 'week', 'cn', 'cv']].merge(same.reset_index(), on=[key, 'season', 'week'], how='left').fillna(0)
    out['cn'] -= out[ncol]; out['cv'] -= out[vcol]
    last = g.groupby([key, 'season'], as_index=False)[[ncol, vcol]].sum(); last['season'] += 1
    out = out[[key, 'season', 'week', 'cn', 'cv']].merge(last, on=[key, 'season'], how='left').fillna(0)
    return out.assign(N=out.cn + prev_w * out[ncol], V=out.cv + prev_w * out[vcol])[[key, 'season', 'week', 'N', 'V']]


def add(df, M):
    df = df.copy()
    path = f'{M}/data/apv_games.parquet'
    df['apv_r'] = 0.0; df['opp_rapv'] = 0.0; df['rook_qb'] = 0.0; df['rook_apv'] = 0.0
    if os.path.exists(path):
        G = pd.read_parquet(path)
        pg = G[G.kind == 'p'].rename(columns={'key': 'pid'})
        t = _to_date(df, pg, 'pid', 'n_rush', 'v_rush', 1.0)
        df = df.merge(t.rename(columns={'N': '_rn', 'V': '_rv'}), on=['pid', 'season', 'week'], how='left')
        df['apv_r'] = (df._rv.fillna(0) / (df._rn.fillna(0) + K_RUN))
        dg = G[G.kind == 'd'].rename(columns={'key': 'opp'})
        t = _to_date(df, dg, 'opp', 'd_n', 'd_adj', 0.5)
        df = df.merge(t.rename(columns={'N': '_dn', 'V': '_dv'}), on=['opp', 'season', 'week'], how='left')
        df['opp_rapv'] = (df._dv.fillna(0) / (df._dn.fillna(0) + K_OPP))
        df = df.drop(columns=['_rn', '_rv', '_dn', '_dv'])
    cp = f'{HERE}/data/qb_college_apv.csv'
    if os.path.exists(cp):
        Q = pd.read_csv(cp)
        ap = dict(zip(Q.gsis_id, Q.col_APV)); dy = dict(zip(Q.gsis_id, Q.season))
        rook = (df.position == 'QB') & (df.season == df.pid.map(dy))
        v = (df.pid.map(ap) - ROOK_CENTER) / 10
        df['rook_qb'] = rook.astype(float)
        df['rook_apv'] = np.where(rook, v.fillna(0), 0.0)
    return df


if __name__ == '__main__':
    dl, M = sys.argv[1], sys.argv[2]
    pos = pd.read_parquet(f'{M}/data/player_games.parquet', columns=['pid', 'position']).drop_duplicates('pid', keep='last')
    G = build_games(dl, dict(zip(pos.pid, pos.position)))
    G.to_parquet(f'{M}/data/apv_games.parquet', index=False)
    print(G.groupby(['kind', 'season']).size().unstack(0).tail(3))
