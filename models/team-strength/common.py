"""Shared pieces for the team-strength models: point-in-time ratings, market-implied ratings, metrics.

Every rating here is computed walk-forward: the value attached to a game uses only games played before it.
"""
import numpy as np, pandas as pd
from scipy.stats import norm

def margin_ratings(g, k=0.1, hfa=2.0, carry=0.6, key='season', home='home', away='away', margin='margin',
                   neutral=None, out='rt', mean_reg=0.0, priors=None, finals=None):
    """Kalman-lite margin rating. pred = R[home] - R[away] + hfa; after the game both teams move k*(error).
    Between seasons every rating is multiplied by `carry` (regression to the mean), unless `priors` ({season: {team:
    rating}}) gives a preseason rating for that team. `finals`, if a dict, is filled with {season: {team: end rating}}.
    Adds columns {out}_home, {out}_away, {out}_pred (pre-game) to a copy of g (must be sorted by date)."""
    R = {}; last = None
    ph, pa, pp = np.empty(len(g)), np.empty(len(g)), np.empty(len(g))
    H, A, M, S = g[home].values, g[away].values, g[margin].values, g[key].values
    N = g[neutral].values if neutral else np.zeros(len(g), bool)
    for i in range(len(g)):
        if S[i] != last:
            if finals is not None and last is not None: finals[last] = dict(R)
            R = {t: carry*r for t, r in R.items()}; last = S[i]
            if priors is not None and S[i] in priors: R.update(priors[S[i]])
        rh, ra = R.get(H[i], 0.0), R.get(A[i], 0.0)
        pred = rh - ra + (0 if N[i] else hfa)
        ph[i], pa[i], pp[i] = rh, ra, pred
        if not np.isnan(M[i]):
            e = M[i] - pred
            R[H[i]] = rh + k*e; R[A[i]] = ra - k*e
    if finals is not None and last is not None: finals[last] = dict(R)
    g = g.copy(); g[out+'_home'] = ph; g[out+'_away'] = pa; g[out+'_pred'] = pp
    return g

def market_ratings(g, line_col, k=0.25, carry=0.8, hfa=None, key='season', home='home', away='away', neutral=None, out='mkt'):
    """Same filter, but it learns from the CLOSING LINE (expected margin) instead of the final score.
    The rating then says what the market thinks of each team before the next game is priced.
    line_col must be the expected home margin (positive = home favoured)."""
    if hfa is None: hfa = g[line_col].mean()
    return margin_ratings(g, k=k, hfa=hfa, carry=carry, key=key, home=home, away=away, margin=line_col, neutral=neutral, out=out)

def tune(g, fn, grid, target='margin', pred_col='rt_pred', mask=None):
    best = None
    for params in grid:
        h = fn(g, **params)
        m = h[mask(h)] if mask is not None else h
        mse = np.nanmean((m[target] - m[pred_col])**2)
        if best is None or mse < best[0]: best = (mse, params)
    return best

def grid(**axes):
    import itertools
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]

def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6); return -np.mean(y*np.log(p) + (1 - y)*np.log(1 - p))

def brier(p, y): return np.mean((p - y)**2)

def win_prob_from_margin(pred, sigma): return norm.cdf(pred / sigma)

def ats_record(pick_home, margin, line):
    """pick_home: bool array (True=home side). line = expected home margin (home -3 => 3). Returns wins, losses, pushes."""
    cover = margin - line
    w = np.where(pick_home, cover > 0, cover < 0); l = np.where(pick_home, cover < 0, cover > 0)
    return int(w.sum()), int(l.sum()), int((cover == 0).sum())

def binom_p(w, n, p0=0.5):
    from scipy.stats import binomtest
    return binomtest(w, n, p0, alternative='greater').pvalue if n else np.nan

def od_ratings(rows, k=0.05, carry=0.6, out='x'):
    """Offense/defense filter on a per-team-game stat (e.g. EPA/play).
    rows: DataFrame sorted by date with columns season, team, opp, value (one row per team per game; both sides present).
    Model: value = mu_season + O[team] + D[opp]. Pre-game O/D are attached; then both update by k*error.
    D is 'points allowed' style: higher D = worse defense."""
    O, D = {}, {}; last = None; mu = rows['value'].mean()
    po, pd_ = np.empty(len(rows)), np.empty(len(rows))
    S, T, P, V = rows.season.values, rows.team.values, rows.opp.values, rows['value'].values
    for i in range(len(rows)):
        if S[i] != last:
            O = {t: carry*r for t, r in O.items()}; D = {t: carry*r for t, r in D.items()}; last = S[i]
        o, d = O.get(T[i], 0.0), D.get(P[i], 0.0)
        po[i], pd_[i] = o, d
    # second pass with updates in game order (both team-rows of a game use pre-game values, so update after the pair)
    O, D = {}, {}; last = None
    gid = rows.game_id.values
    i = 0; n = len(rows)
    while i < n:
        j = i
        while j < n and gid[j] == gid[i]: j += 1
        if S[i] != last:
            O = {t: carry*r for t, r in O.items()}; D = {t: carry*r for t, r in D.items()}; last = S[i]
        pre = [(O.get(T[m], 0.0), D.get(P[m], 0.0)) for m in range(i, j)]
        for m, (o, d) in zip(range(i, j), pre):
            po[m], pd_[m] = o, d
        for m, (o, d) in zip(range(i, j), pre):
            if not np.isnan(V[m]):
                e = V[m] - (mu + o + d)
                O[T[m]] = O.get(T[m], 0.0) + k*e; D[P[m]] = D.get(P[m], 0.0) + k*e
        i = j
    r = rows.copy(); r[out+'_O'] = po; r[out+'_Dopp'] = pd_
    return r
