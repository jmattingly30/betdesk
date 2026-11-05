"""Opponent-adjusted tempo-and-efficiency ratings (KenPom/Torvik style), computed walk-forward.

For every team-game (offense i vs defense j):   100*pts/poss = mu + O_i - D_j + hca*loc     (loc +1 home, -1 away, 0 neutral)
For every game:                                 poss          = T0 + P_home + P_away
Both are ridge regressions on the season's games played BEFORE the prediction date, with recency weights and a prior
pulling each team toward its preseason rating (last season's final rating x carry, plus a returning-minutes shift when
available). The prior counts as `prior_games` games and fades as real games accumulate.

Prediction for a game:  poss_hat = T0 + P_h + P_a
                         home_ppp = mu + O_h - D_a + hca(if not neutral)      away_ppp = mu + O_a - D_h - hca
                         home_pts = home_ppp*poss_hat/100, away_pts likewise; margin and total follow.
"""
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr

DEFAULTS = dict(prior_games=6.0, carry=0.65, tempo_carry=0.75, half_life=60.0, new_team_o=-6.0, new_team_d=6.0,
                ret_coef=0.0, cap=None, cap_slope=0.5)
# 2026-10-09 audit additions:
#   cap/cap_slope: blowout dampening. A game's efficiency margin (points per 100, team minus opponent) beyond `cap` counts
#                  only `cap_slope` per point, so a 45-point rout of a bad team moves ratings less (KenPom/Torvik style).
#   g['loc_h']:    optional home-team location value replacing +1/0 (e.g. a 'neutral' game in the home-listed team's
#                  state = partial home edge). Away team gets -loc_h.

def _fit(rows_i, rows_j, loc, y, w, n, prior_vals, prior_w, kind):
    """Weighted ridge-to-prior least squares. kind='eff': params O(n), D(n), hca, mu. kind='pace': P(n), T0."""
    m = len(y)
    if kind == 'eff':
        r = np.arange(m)
        A = csr_matrix((np.r_[np.ones(m), -np.ones(m), loc, np.ones(m)], (np.r_[r, r, r, r], np.r_[rows_i, n + rows_j, np.full(m, 2*n), np.full(m, 2*n + 1)])), shape=(m, 2*n + 2))
        k = 2*n + 2
    else:
        r = np.arange(m)
        A = csr_matrix((np.r_[np.ones(m), np.ones(m), np.ones(m)], (np.r_[r, r, r], np.r_[rows_i, rows_j, np.full(m, n)])), shape=(m, n + 1))
        k = n + 1
    sw = np.sqrt(w)
    A = csr_matrix(A.multiply(sw[:, None])); b = y * sw
    npar = len(prior_vals)
    P = csr_matrix((np.sqrt(prior_w), (np.arange(npar), np.arange(npar))), shape=(npar, k))
    A = vstack([A, P]); b = np.r_[b, np.sqrt(prior_w) * prior_vals]
    if kind == 'eff':  # weak anchors for hca and mu so they never run away early in the season
        anc = csr_matrix(([3.0, 0.3], ([0, 1], [2*n, 2*n + 1])), shape=(2, k)); A = vstack([A, anc]); b = np.r_[b, 3.0 * 1.6, 0.3 * 103.0]
    return lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=2000)[0]

def walk_forward(g, tg, ret=None, priors=None, verbose=False, **kw):
    """g: games (one row per game, sorted by date, D1 only); tg: team-games for those games.
    ret: optional DataFrame [season, team, ret_min] returning-minutes share for the preseason prior.
    priors: optional {season: DataFrame(index=team, columns O, D, P)} preseason ratings that replace the carry rule.
    Returns g with pre-game ratings and predictions, plus end-of-season ratings per team."""
    p = {**DEFAULTS, **kw}
    g = g.sort_values(['date', 'game_id']).reset_index(drop=True)
    tg = tg[tg.game_id.isin(g.game_id)].copy()
    tg['ppp'] = 100 * tg.pts / tg.poss
    if 'loc_h' in g.columns:
        lh = g.set_index('game_id').loc_h
        tg['loc'] = np.where(tg.ha == 'home', 1, -1) * lh.reindex(tg.game_id).values
    else:
        tg['loc'] = np.where(tg.neutral, 0, np.where(tg.ha == 'home', 1, -1))
    if p['cap'] is not None:
        opp = tg.set_index(['game_id', 'team']).ppp
        m = tg.ppp.values - opp.reindex(pd.MultiIndex.from_arrays([tg.game_id, tg.opp])).values
        a = np.abs(m); m2 = np.sign(m) * np.where(a > p['cap'], p['cap'] + p['cap_slope'] * (a - p['cap']), a)
        tg['ppp'] = tg.ppp + np.nan_to_num((m2 - m) / 2)
    out_cols = {c: np.full(len(g), np.nan) for c in ['O_h', 'D_h', 'O_a', 'D_a', 'P_h', 'P_a', 'mu', 'hca', 'T0', 'n_h', 'n_a']}
    final = {}; prev = None
    for season, gs in g.groupby('season', sort=True):
        teams = np.unique(np.r_[gs.home.values, gs.away.values]); n = len(teams); idx = {t: i for i, t in enumerate(teams)}
        # preseason prior
        pO = np.full(n, p['new_team_o']); pD = np.full(n, p['new_team_d']); pP = np.zeros(n)
        if prev is not None:
            r = ret.set_index('team').ret_min[ret.season == season] if ret is not None else None
            for t, i in idx.items():
                if t in prev.index:
                    o_, d_, pc = prev.loc[t, ['O', 'D', 'P']]
                    shift = 0.0
                    if r is not None and t in r.index and not np.isnan(r[t]):
                        shift = p['ret_coef'] * (r[t] - 0.55)   # more returning minutes -> better than last-year-regressed
                    c = p['carry']
                    pO[i], pD[i], pP[i] = c*o_ + shift/2, c*d_ + shift/2, p['tempo_carry']*pc
        if priors is not None and season in priors:
            pr = priors[season]
            for t, i in idx.items():
                if t in pr.index: pO[i], pD[i], pP[i] = pr.loc[t, ['O', 'D', 'P']]
        tgs = tg[tg.season == season]
        ti = tgs.team.map(idx).values; tj = tgs.opp.map(idx).values; days = tgs.date.values
        gi_h = gs.home.map(idx).values; gi_a = gs.away.map(idx).values
        gpos = gs.poss.values; gdays = gs.date.values
        dates = np.unique(gs.date.values)
        est = None
        for dt in dates:
            mask = days < dt
            if mask.any():
                age = (dt - days[mask]).astype('timedelta64[D]').astype(float)
                w = 0.5 ** (age / p['half_life'])
                eff = _fit(ti[mask], tj[mask], tgs['loc'].values[mask], tgs.ppp.values[mask], w, n,
                           np.r_[pO, pD], np.full(2*n, p['prior_games']), 'eff')
                gm = gdays < dt; ageg = (dt - gdays[gm]).astype('timedelta64[D]').astype(float)
                pace = _fit(gi_h[gm], gi_a[gm], None, gpos[gm], 0.5 ** (ageg / p['half_life']), n, pP, np.full(n, p['prior_games']), 'pace')
                est = (eff[:n], eff[n:2*n], eff[2*n], eff[2*n + 1], pace[:n], pace[n])
            else:
                est = (pO, pD, 1.6, 103.0 if prev is None else prev.attrs['mu'], pP, 68.0 if prev is None else prev.attrs['T0'])
            O, D, hca, mu, P, T0 = est
            sel = np.where(gdays == dt)[0]
            gl = gs.index.values[sel]
            h, a = gi_h[sel], gi_a[sel]
            for col, v in [('O_h', O[h]), ('D_h', D[h]), ('O_a', O[a]), ('D_a', D[a]), ('P_h', P[h]), ('P_a', P[a])]: out_cols[col][gl] = v
            out_cols['mu'][gl] = mu; out_cols['hca'][gl] = hca; out_cols['T0'][gl] = T0
        # end-of-season ratings using every game
        w = 0.5 ** ((days.max() - days).astype('timedelta64[D]').astype(float) / p['half_life'])
        eff = _fit(ti, tj, tgs['loc'].values, tgs.ppp.values, w, n, np.r_[pO, pD], np.full(2*n, p['prior_games']), 'eff')
        wg = 0.5 ** ((gdays.max() - gdays).astype('timedelta64[D]').astype(float) / p['half_life'])
        pace = _fit(gi_h, gi_a, None, gpos, wg, n, pP, np.full(n, p['prior_games']), 'pace')
        # centre so that O, D and P are relative to the average D1 team (mu, T0 absorb the shift)
        O, D, P = eff[:n], eff[n:2*n], pace[:n]
        mu = eff[2*n + 1] + O.mean() - D.mean(); T0 = pace[n] + 2*P.mean()
        O, D, P = O - O.mean(), D - D.mean(), P - P.mean()
        prev = pd.DataFrame({'O': O, 'D': D, 'P': P}, index=teams)
        prev.attrs['mu'] = mu; prev.attrs['T0'] = T0
        final[season] = prev.assign(net=O + D, mu=mu, hca=eff[2*n], T0=T0)
        if verbose: print(season, 'mu %.1f hca %.2f T0 %.1f' % (mu, eff[2*n], T0), flush=True)
    for c, v in out_cols.items(): g[c] = v
    hc = g.hca * g.loc_h if 'loc_h' in g.columns else np.where(g.neutral, 0, g.hca)
    g['poss_hat'] = g.T0 + g.P_h + g.P_a
    g['ppp_h'] = g.mu + g.O_h - g.D_a + hc; g['ppp_a'] = g.mu + g.O_a - g.D_h - hc
    g['pts_h_hat'] = g.ppp_h * g.poss_hat / 100; g['pts_a_hat'] = g.ppp_a * g.poss_hat / 100
    g['margin_hat'] = g.pts_h_hat - g.pts_a_hat; g['total_hat'] = g.pts_h_hat + g.pts_a_hat
    return g, final
