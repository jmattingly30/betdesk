"""CFB v3: rebuilt college football side and total model (2026-10-09, J asked for a rebuild from what the project has).

What J's v2 had that the public team-strength model lacked: opponent-adjusted play-level efficiency (v2 used CFBD PPA,
a ridge regression on play-level data). CFBD's API is blocked from the cloud, so v3 uses ESPN play-by-play EPA (the same
idea: expected points added per play, from sportsdataverse) and refits, every week, walk-forward:
  1. margin ridge:  margin = hfa*home + R[home] - R[away]
  2. efficiency ridge (non-garbage EPA/play, one row per offense): epa = mu + h*home + Off[team] - Def[opp]
  3. points ridge (for totals): pts = mu + h*home + Op[team] + Dp[opp]
Each is a ridge toward a preseason prior: last season's final rating x carry, plus returning offensive production.
Current season games weigh 1, last season's weigh `w_prev`. A stack (fit on 2014-2019, 2021) turns the three into a
margin and a total; 2022-2025 is the holdout, judged against CFBD opening and closing lines.

usage: python3 build.py <cfb_games.parquet> <epa.csv> <returning.csv> <outdir> [pbp_2026.parquet sched_2026.csv]"""
import sys, os, json, numpy as np, pandas as pd
# CFB-2 (2026-10-10): --roster roster_prior.csv adds roster-built inputs (transfer-aware QB and skill production, from
# roster_prior.py) to the margin and total stacks
ROSTER = None
if '--roster' in sys.argv:
    i = sys.argv.index('--roster'); ROSTER = sys.argv[i + 1]; del sys.argv[i:i + 2]
games_pq, epa_csv, ret_csv, outdir = sys.argv[1:5]
os.makedirs(outdir, exist_ok=True)
G = pd.read_parquet(games_pq)
G = G[G.home_points.notna() & G.away_points.notna() & (G.season >= 2013)].copy()
E = pd.read_csv(epa_csv)
if len(sys.argv) > 6:   # add 2026 from ESPN play-by-play + schedule (scores, ESPN spread)
    import pyarrow.parquet as pq
    pb, sc = sys.argv[5], sys.argv[6]
    cols = ['season', 'game_id', 'pos_team_id', 'def_pos_team_id', 'EPA', 'EPA_success', 'wp_before', 'scrimmage_play', 'pass', 'rush',
            'homeTeamId', 'homeFinalScore', 'awayFinalScore', 'homeTeamSpread', 'homeFavorite', 'week']
    p = pq.read_table(pb, columns=cols).to_pandas()
    fin = p.groupby('game_id').agg(hf=('homeFinalScore', 'max'), af=('awayFinalScore', 'max'), sp=('homeTeamSpread', 'first'), fav=('homeFavorite', 'first'))
    s = pd.read_csv(sc); s = s[s.game_id.isin(fin.index) | s.completed]
    s = s.join(fin, on='game_id')
    s['home_points'] = s.home_points.fillna(s.hf); s['away_points'] = s.away_points.fillna(s.af)
    s['espn_line'] = np.where(s.fav == True, s.sp.abs(), -s.sp.abs())   # expected home margin
    s = s[s.home_points.notna()]
    s['neutral'] = s.neutral_site.astype(bool); s['date'] = pd.to_datetime(s.start_date, utc=True)
    s['line_open'] = np.nan; s['line_close'] = np.nan; s['total_open'] = np.nan; s['total_close'] = np.nan
    G = pd.concat([G[G.season < 2026], s[[c for c in G.columns if c in s.columns] + ['espn_line']]], ignore_index=True)
    q = p[(p.scrimmage_play == True) & p.EPA.notna()].copy(); q['ng'] = q.wp_before.between(0.1, 0.9)
    k = ['season', 'game_id', 'pos_team_id', 'def_pos_team_id']; gb = q.groupby(k)
    e26 = pd.DataFrame({'plays': gb.size(), 'epa': gb.EPA.mean(), 'sr': gb.EPA_success.mean(), 'ng_epa': q[q.ng].groupby(k).EPA.mean()}).reset_index()
    E = pd.concat([E[E.season < 2026], e26], ignore_index=True)
G['date'] = pd.to_datetime(G.date, utc=True); G = G.sort_values(['date', 'game_id']).reset_index(drop=True)
G['margin'] = G.home_points - G.away_points; G['total'] = G.home_points + G.away_points
G['neutral'] = G.neutral.fillna(False).astype(bool)
G['wk'] = np.where(G.season_type.eq('postseason'), 20, G.week)
R = pd.read_csv(ret_csv)
fbs = set(G[G.home_division.eq('fbs')].home_id)

def ridge(rows, cols_idx, y, w, lam, prior, ncol, extra_cols=0):
    """rows: list of (i, list[(col, val)]); solve weighted ridge for deviation from prior."""
    n = len(y); X = np.zeros((n, ncol + extra_cols))
    for r, ent in enumerate(rows):
        for c, v in ent: X[r, c] += v
    yy = y - X[:, :ncol] @ prior
    sw = np.sqrt(w); A = X * sw[:, None]; b = yy * sw
    P = np.eye(ncol + extra_cols) * lam
    for j in range(ncol, ncol + extra_cols): P[j, j] = 1e-3
    d = np.linalg.solve(A.T @ A + P, A.T @ b)
    full = np.concatenate([prior, np.zeros(extra_cols)]) + d
    return full

MK_PAR = dict(k=0.5, carry=0.85, hfa=3.0)   # market rating filter, tuned 2014-2021 on predicting the next closing line
PAR = dict(lam_m=0.5, lam_e=25.0, lam_p=1.0, carry=0.6, w_prev=0.15, ret_m=6.0, ret_e=0.08, ret_p=3.0)
def run(par, seasons):
    """walk-forward weekly ratings; returns per-game pre-game features."""
    feats = []; finals = {}
    for s in seasons:
        gs = G[G.season == s]; gp = G[G.season == s - 1]
        es = E[E.season == s]; ep = E[E.season == s - 1]
        teams = sorted(set(gs.home_id) | set(gs.away_id) | set(gp.home_id) | set(gp.away_id))
        ix = {t: i for i, t in enumerate(teams)}; n = len(teams)
        rt = R[R.season == s].set_index('team_id').ret_off; rmean = rt.mean() if len(rt) else 0.5
        pf = finals.get(s - 1, {})
        def pri(key, retc):
            v = np.zeros(n)
            for t, i in ix.items():
                base = par['carry'] * pf.get(key, {}).get(t, 0.0)
                v[i] = base + retc * (rt.get(t, rmean) - rmean)
            return v
        pm, po, pdf, pop, pdp = pri('m', par['ret_m']), pri('o', par['ret_e']), pri('d', 0.0), pri('op', par['ret_p']), pri('dp', 0.0)
        weeks = sorted(gs.wk.unique())
        for wk in weeks + [99]:
            tr = pd.concat([gp.assign(_w=par['w_prev']), gs[gs.wk < wk].assign(_w=1.0)])
            # margin
            rows = [[(ix[h], 1), (ix[a], -1), (n, 0 if nu else 1)] for h, a, nu in zip(tr.home_id, tr.away_id, tr.neutral)]
            fm = ridge(rows, None, tr.margin.values.astype(float), tr._w.values, par['lam_m'], pm, n, 1)
            # points (two rows per game)
            rp, yp, wp = [], [], []
            for h, a, nu, hp, ap_, w in zip(tr.home_id, tr.away_id, tr.neutral, tr.home_points, tr.away_points, tr._w):
                rp.append([(ix[h], 1), (n + ix[a], 1), (2 * n, 1), (2 * n + 1, 0 if nu else 1)]); yp.append(hp); wp.append(w)
                rp.append([(ix[a], 1), (n + ix[h], 1), (2 * n, 1), (2 * n + 1, 0 if nu else -1)]); yp.append(ap_); wp.append(w)
            fp = ridge(rp, None, np.array(yp, float), np.array(wp), par['lam_p'], np.concatenate([pop, pdp]), 2 * n, 2)
            # efficiency
            et = pd.concat([ep.assign(_w=par['w_prev']), es[es.game_id.isin(gs[gs.wk < wk].game_id)].assign(_w=1.0)])
            et = et[et.pos_team_id.isin(ix) & et.def_pos_team_id.isin(ix) & et.ng_epa.notna()]
            re_ = [[(ix[o], 1), (n + ix[d], -1), (2 * n, 1)] for o, d in zip(et.pos_team_id, et.def_pos_team_id)]
            fe = ridge(re_, None, et.ng_epa.values, et._w.values * np.sqrt(et.plays.clip(upper=80).values / 60), par['lam_e'], np.concatenate([po, pdf]), 2 * n, 1) if len(et) else np.concatenate([po, pdf, [0]])
            if wk == 99:
                finals[s] = {'m': {t: fm[i] for t, i in ix.items()}, 'o': {t: fe[i] for t, i in ix.items()}, 'd': {t: fe[n + i] for t, i in ix.items()},
                             'op': {t: fp[i] for t, i in ix.items()}, 'dp': {t: fp[n + i] for t, i in ix.items()},
                             'hfa': fm[n], 'mu_p': fp[2 * n], 'h_p': fp[2 * n + 1], 'mu_e': fe[2 * n] if len(fe) > 2 * n else 0}
                continue
            cur = gs[gs.wk == wk]
            for r in cur.itertuples():
                h, a = ix[r.home_id], ix[r.away_id]; hf = 0 if r.neutral else 1
                m_m = fm[h] - fm[a] + hf * fm[n]
                m_e = (fe[h] - fe[n + a]) - (fe[a] - fe[n + h])
                ph = fp[2 * n] + fp[h] + fp[n + a] + hf * fp[2 * n + 1]; pa = fp[2 * n] + fp[a] + fp[n + h] - hf * fp[2 * n + 1]
                feats.append(dict(game_id=r.game_id, m_m=m_m, m_e=m_e, m_p=ph - pa, t_p=ph + pa,
                                  o_sum=fe[h] + fe[a] - fe[n + h] - fe[n + a], hf=hf))
    return pd.DataFrame(feats), finals

if __name__ == '__main__':
    if '--tune' in sys.argv:
        from numpy.linalg import lstsq
        for lm, le, lp in [(0.5, 5, 1), (1.5, 10, 2), (3, 25, 4), (0.5, 25, 1)]:
            par = dict(PAR, lam_m=lm, lam_e=le, lam_p=lp)
            F, _ = run(par, range(2014, 2020)); D = G.merge(F, on='game_id'); D = D[D.season >= 2015]
            X = np.column_stack([D.m_m, D.m_e, D.m_p, D.hf, np.ones(len(D))]); b = lstsq(X, D.margin, rcond=None)[0]
            Xt = np.column_stack([D.t_p, D.o_sum, np.ones(len(D))]); bt_ = lstsq(Xt, D.total, rcond=None)[0]
            print(lm, le, lp, 'margin MAE %.3f total MAE %.3f' % ((D.margin - X @ b).abs().mean(), (D.total - Xt @ bt_).abs().mean()), np.round(b, 2), flush=True)
        sys.exit()
    F, finals = run(PAR, range(2014, int(G.season.max()) + 1))
    D = G.merge(F, on='game_id')
    D['elo_d'] = (D.home_pregame_elo - D.away_pregame_elo) / 25.0
    D['both_fbs'] = D.home_id.isin(fbs) & D.away_id.isin(fbs)
    TR = D.season.between(2014, 2021) & (D.season != 2020); TE = D.season.between(2022, 2025)
    # margin stack: separate fits for weeks 1-3 (prior-heavy) and 4+; CFBD pregame Elo where present
    D['elo_ok'] = D.elo_d.notna().astype(float); D['elo_d'] = D.elo_d.fillna(0)
    cols = ['m_m', 'm_e', 'hf', 'elo_d', 'elo_ok']
    # market rating (J 2026-10-26, season-adjustments card 1): each team's rating learned walk-forward from its past
    # closing lines (2026: ESPN's game-time line), never the current game's. v3 alone stayed stuck on teams the market
    # had moved off for weeks; with this input the stack learns how much of that market view to trust.
    sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'team-strength')))
    from common import margin_ratings
    G['mk_line'] = G.line_close if 'espn_line' not in G else G.line_close.fillna(G.espn_line)
    MK = margin_ratings(G, k=MK_PAR['k'], hfa=MK_PAR['hfa'], carry=MK_PAR['carry'], home='home_id', away='away_id', margin='mk_line',
                        neutral='neutral', out='mk', finals=(mk_finals := {}))
    D = D.merge(MK[['game_id', 'mk_pred']].rename(columns={'mk_pred': 'mk'}), on='game_id', how='left'); D['mk'] = D.mk.fillna(0.0)
    cols += ['mk']
    ct_extra = []
    if ROSTER:
        RP = pd.read_csv(ROSTER); RP['qb_q'] = RP.qb_anya + 0.3 * RP.qb_lvl * (1 - RP.qb_none)
        rk = ['qb_q', 'qb_new', 'qb_none', 'avail_skill', 'xfer_skill']
        for side in ['home', 'away']:
            D = D.merge(RP[['season', 'team_id'] + rk].rename(columns={'team_id': f'{side}_id', **{c: f'{side}_{c}' for c in rk}}), on=['season', f'{side}_id'], how='left')
        dflt = dict(qb_q=0.0, qb_new=0, qb_none=1, avail_skill=0.3, xfer_skill=0.0)
        for c in rk:
            for side in ['home', 'away']: D[f'{side}_{c}'] = D[f'{side}_{c}'].fillna(dflt[c])
            D[f'r_{c}'] = D[f'home_{c}'] - D[f'away_{c}']
        cols += [f'r_{c}' for c in rk]
        # tried in the total stack too (rt_qb, rt_skill): holdout total MAE 12.93 -> 12.97, so totals keep the v3 inputs only
    X = lambda d: np.column_stack([d[c] for c in cols] + [np.ones(len(d))])
    from numpy.linalg import lstsq
    early = D.wk <= 3
    # stacks are fit on games that had a betting line (the games the app prices); 2026-10-26: needed for the market rating,
    # which carries no information for the many FCS-vs-FCS games that never get a line
    lined = D.line_close.notna()
    bm = {k: lstsq(X(D[TR & msk & lined]), D.margin[TR & msk & lined], rcond=None)[0] for k, msk in [('early', early), ('late', ~early)]}
    D['model_margin'] = np.where(early, X(D) @ bm['early'], X(D) @ bm['late'])
    # mismatch term (J 2026-10-26, season-adjustments card 5): totals ran 1-6 pts low in 24+ pt mismatches every season
    # 2018-2026 and 1-2 high in close games, so the total stack gets the size of the predicted margin (and its square)
    D['am'] = D.model_margin.abs(); D['am2'] = D.model_margin ** 2 / 100.0
    ct_extra += ['am', 'am2']
    ct = ['t_p', 'o_sum'] + ct_extra
    bt = lstsq(np.column_stack([D[TR][ct], np.ones(TR.sum())]), D.total[TR], rcond=None)[0]
    D['model_total'] = np.column_stack([D[ct], np.ones(len(D))]) @ bt
    # opener blend weights (fit on train, applied on holdout)
    def blend(d, line, model, y):
        m = d[line].notna(); X_ = np.column_stack([d[line][m], d[model][m] - d[line][m]]); return lstsq(X_, d[y][m], rcond=None)[0]
    tro = D[TR & D.line_open.notna() & ((D.line_open - D.line_close).abs() <= 10)]
    b_open = blend(tro, 'line_open', 'model_margin', 'margin'); b_close = blend(D[TR], 'line_close', 'model_margin', 'margin')
    bt_open = blend(D[TR & D.total_open.notna()], 'total_open', 'model_total', 'total'); bt_close = blend(D[TR], 'total_close', 'model_total', 'total')
    log = []
    def P(*a): s = ' '.join(str(x) for x in a); print(s); log.append(s)
    P('stack margin coefs', {k: dict(zip(cols + ['const'], np.round(v, 3))) for k, v in bm.items()}); P('stack total coefs', dict(zip(ct + ['const'], np.round(bt, 3))))
    P('gap weight b (share of model-minus-line that is real): spread open %.3f close %.3f, total open %.3f close %.3f' % (b_open[1], b_close[1], bt_open[1], bt_close[1]))
    H = D[TE & D.line_open.notna() & D.line_close.notna() & ((D.line_open - D.line_close).abs() <= 10)].copy()
    P(f'\nHOLDOUT 2022-2025, games with open and close lines: {len(H)}')
    P('margin MAE: open %.2f close %.2f model %.2f | public team-strength 14.23 wk1-3 FBS (not comparable subset)' % (
        (H.margin - H.line_open).abs().mean(), (H.margin - H.line_close).abs().mean(), (H.margin - H.model_margin).abs().mean()))
    for lab, msk in [('weeks 1-3', H.wk <= 3), ('weeks 4-6', H.wk.between(4, 6)), ('weeks 4+', H.wk > 3), ('FBS v FBS', H.both_fbs)]:
        h = H[msk]; P(f'  {lab} n {len(h)}: open %.2f close %.2f model %.2f' % ((h.margin - h.line_open).abs().mean(), (h.margin - h.line_close).abs().mean(), (h.margin - h.model_margin).abs().mean()))
    from scipy.stats import binomtest
    rows = []
    for thr in [2, 3, 4, 5, 6, 8]:
        for at in ['open', 'close']:
            L = H[f'line_{at}']; gap = H.model_margin - L; pick = gap.abs() >= thr
            res = (H.margin - L)[pick] * np.sign(gap[pick]); w, l = int((res > 0).sum()), int((res < 0).sum())
            clv = ((H.line_close - H.line_open) * np.sign(gap))[pick] if at == 'open' else pd.Series(dtype=float)
            rows.append(dict(market='spread', at=at, gap=thr, bets=w + l, win=round(w / max(w + l, 1), 3), p=round(binomtest(w, w + l, 0.524, 'greater').pvalue, 3) if w + l else None,
                             clv_pts=round(clv.mean(), 2) if len(clv) else None, clv_t=round(clv.mean() / clv.std() * np.sqrt(len(clv)), 1) if len(clv) > 2 else None))
    HT = D[TE & D.total_open.notna() & D.total_close.notna() & ((D.total_open - D.total_close).abs() <= 10)]
    for thr in [3, 5, 7, 10]:
        for at in ['open', 'close']:
            L = HT[f'total_{at}']; gap = HT.model_total - L; pick = gap.abs() >= thr
            res = (HT.total - L)[pick] * np.sign(gap[pick]); w, l = int((res > 0).sum()), int((res < 0).sum())
            clv = ((HT.total_close - HT.total_open) * np.sign(gap))[pick] if at == 'open' else pd.Series(dtype=float)
            rows.append(dict(market='total', at=at, gap=thr, bets=w + l, win=round(w / max(w + l, 1), 3), p=round(binomtest(w, w + l, 0.524, 'greater').pvalue, 3) if w + l else None,
                             clv_pts=round(clv.mean(), 2) if len(clv) else None, clv_t=round(clv.mean() / clv.std() * np.sqrt(len(clv)), 1) if len(clv) > 2 else None))
    for lab, msk in [('weeks 1-4', H.wk <= 4), ('weeks 5+', H.wk > 4)]:
        h = H[msk]; gap = h.model_margin - h.line_open; pk = gap.abs() >= 2; res = ((h.margin - h.line_open) * np.sign(gap))[pk]
        clv = ((h.line_close - h.line_open) * np.sign(gap))[pk]
        P(f'  spread vs opener, 2+ pt gaps, {lab}: n {int((res != 0).sum())} win {(res > 0).sum() / max((res != 0).sum(), 1):.3f} CLV {clv.mean():+.2f} (t {clv.mean() / clv.std() * np.sqrt(len(clv)):.1f})')
    T = pd.DataFrame(rows); P('\n' + T.to_string(index=False)); T.to_csv(f'{outdir}/holdout_bets.csv', index=False)
    P('total MAE holdout: open %.2f close %.2f model %.2f' % ((HT.total - HT.total_open).abs().mean(), (HT.total - HT.total_close).abs().mean(), (HT.total - HT.model_total).abs().mean()))
    # residual sds for probabilities
    sd_m = float((D[TE & lined].margin - D[TE & lined].model_margin).std()); sd_t = float((D[TE].total - D[TE].model_total).std())
    # win chance check (J 2026-10-26, season-adjustments card 2): underdog chances ran high because sd_margin was measured
    # on every game, FCS mismatches included (17-19 pts). Measured on games with a line it is 15.6, and the plain bell
    # curve then matches a fitted curve P = Phi(a*m + b*m*|m|/100) (holdout log loss equal), so the app keeps the bell
    # curve at this sd; the fitted curve is logged here as the check
    from scipy.stats import norm as _N
    from scipy.optimize import minimize as _min
    W = D[TR & lined & (D.margin != 0)]; wy = (W.margin > 0).astype(float).values; wm = W.model_margin.values
    def _nll(th):
        q = np.clip(_N.cdf(th[0] * wm + th[1] * wm * np.abs(wm) / 100), 1e-6, 1 - 1e-6); return -np.mean(wy * np.log(q) + (1 - wy) * np.log(1 - q))
    win_ab = _min(_nll, [1 / 16.0, 0.0], method='Nelder-Mead').x
    WH = D[TE & lined & (D.margin != 0)]; hy = (WH.margin > 0).astype(float).values; hm = WH.model_margin.values
    def _ll(q): q = np.clip(q, 1e-6, 1 - 1e-6); return -np.mean(hy * np.log(q) + (1 - hy) * np.log(1 - q))
    P('win chance curve a %.4f b %.4f; holdout log loss bell (sd %.1f) %.4f, curve %.4f' % (win_ab[0], win_ab[1], sd_m, _ll(_N.cdf(hm / sd_m)), _ll(_N.cdf(win_ab[0] * hm + win_ab[1] * hm * np.abs(hm) / 100))))
    def sd_blend(line, b, model, y, d):
        d = d[d[line].notna()]; return float((d[y] - (d[line] + b * (d[model] - d[line]))).std())
    cfg = dict(PAR, mk_par=MK_PAR, stack_margin={k: dict(zip(cols + ['const'], map(float, v))) for k, v in bm.items()}, stack_total=dict(zip(ct + ['const'], map(float, bt))),
               b_open=float(b_open[1]), b_close=float(b_close[1]), bt_open=float(bt_open[1]), bt_close=float(bt_close[1]),
               sd_margin=sd_m, sd_total=sd_t, sd_open=sd_blend('line_open', b_open[1], 'model_margin', 'margin', D[TE]), sd_close=sd_blend('line_close', b_close[1], 'model_margin', 'margin', D[TE]),
               sdt_open=sd_blend('total_open', bt_open[1], 'model_total', 'total', D[TE]), sdt_close=sd_blend('total_close', bt_close[1], 'model_total', 'total', D[TE]))
    if ROSTER: cfg['roster_file'] = os.path.basename(ROSTER); cfg['roster_defaults'] = dflt
    json.dump(cfg, open(f'{outdir}/model_config.json', 'w'), indent=1)
    D[['game_id', 'season', 'wk', 'home_id', 'away_id', 'neutral', 'margin', 'total', 'line_open', 'line_close', 'total_open', 'total_close', 'model_margin', 'model_total', 't_p', 'mk']].to_parquet(f'{outdir}/game_predictions.parquet', index=False)
    s = int(G.season.max()); f = finals[s]
    names = pd.concat([G[['home_id', 'home_team']].set_axis(['id', 'name'], axis=1), G[['away_id', 'away_team']].set_axis(['id', 'name'], axis=1)]).drop_duplicates('id', keep='last').set_index('id').name
    cur = set(G[G.season == s].home_id) | set(G[G.season == s].away_id)
    rat = pd.DataFrame([dict(team=names.get(t, t), id=t, fbs=t in fbs, m=f['m'][t], o=f['o'][t], d=f['d'][t], op=f['op'][t], dp=f['dp'][t], mk=mk_finals.get(s, {}).get(t, 0.0)) for t in f['m'] if t in cur])
    sl = cfg['stack_margin']['late']; rat['power'] = sl['m_m'] * rat.m + sl['m_e'] * (rat.o + rat.d)
    rat = rat.sort_values('power', ascending=False).round(4); rat.to_csv(f'{outdir}/ratings_current.csv', index=False)
    json.dump(dict(hfa=f['hfa'], mu_p=f['mu_p'], h_p=f['h_p'], season=s, through=str(G[G.season == s].date.max().date())), open(f'{outdir}/ratings_meta.json', 'w'))
    P(f'\ncurrent ratings ({s}, through {G[G.season == s].date.max().date()}):'); P(rat.head(12)[['team', 'power', 'm', 'o', 'd']].to_string(index=False))
    if 'espn_line' in D:
        x = D[(D.season == s) & D.espn_line.notna()]
        if len(x): P(f'\n2026 so far ({len(x)} games with ESPN spread): MAE ESPN line %.2f model %.2f' % ((x.margin - x.espn_line).abs().mean(), (x.margin - x.model_margin).abs().mean()))
    open(f'{outdir}/run_log.txt', 'w').write('\n'.join(log) + '\n')
