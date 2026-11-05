"""Basketball layer 1, weekly re-fit version (the CFB lesson: sparse college schedules reward re-reading early
results once opponents are better known). Every Monday, each component's offense/defense ratings (logit scale;
pace on log scale) are re-solved from all of this season's games so far (+ last season's at weight W_PREV),
shrunk toward a preseason prior (last season's final x carry). That week's games are predicted from those ratings.
Usage: python3 ridge_bb.py sport [params.json] [tag]"""
import sys, os, json, time, numpy as np, pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E

DEFAULT = dict(lam=3.0, lam_pace=3.0, carry=0.6, w_prev=0.1, hl_days=0)

def solve(oi, di, hs, y, w, nT, prior, lam):
    n = len(y); sw = np.sqrt(w)
    ri = np.concatenate([np.arange(n)] * 4)
    ci = np.concatenate([oi, nT + di, np.full(n, 2 * nT), np.full(n, 2 * nT + 1)])
    data = np.concatenate([sw, sw, hs * sw, sw])
    A = csr_matrix((data, (ri, ci)), shape=(n, 2 * nT + 2))
    k = 2 * nT; P = csr_matrix((np.full(k, np.sqrt(lam)), (np.arange(k), np.arange(k))), shape=(k, 2 * nT + 2))
    x = lsqr(vstack([A, P]), np.concatenate([y * sw, np.sqrt(lam) * prior]), atol=1e-7, btol=1e-7, iter_lim=3000)[0]
    return x[:nT], x[nT:2 * nT], x[2 * nT], x[2 * nT + 1]

def run(tg, prm=None, prior_fn=None):
    prm = {**DEFAULT, **(prm or {})}
    g = E.games_from_teamgames(tg)
    teams = pd.unique(pd.concat([g.team, g.team_a])); idx = {t: i for i, t in enumerate(teams)}; nT = len(teams)
    g["hi"] = g.team.map(idx).values; g["ai"] = g.team_a.map(idx).values; g["hs"] = np.where(g.neutral, 0.0, 1.0)
    g["wk"] = (g.date - pd.to_timedelta(g.date.dt.weekday, unit="D")).dt.normalize()
    if prm.get("half_week"): g["wk"] = g.wk + pd.to_timedelta(np.where(g.date.dt.weekday >= 3, 3, 0), unit="D")
    lg = {c: np.average(g["y_" + c].clip(1e-3, 1 - 1e-3), weights=g["n_" + c]) for c in E.COMP}
    for c in E.COMP:
        for s in ("", "_a"): g["l_" + c + s] = E.logit(g["y_" + c + s].clip(1e-3, 1 - 1e-3))
    g["lpace"] = np.log(g.poss)
    K = E.COMP + ["pace"]
    last = {c: (np.zeros(nT), np.zeros(nT)) for c in K}
    rows = []
    for season, gs in g.groupby("season", sort=True):
        prev = g[g.season == season - 1]
        pri = {c: (prm["carry"] * last[c][0], prm["carry"] * last[c][1]) for c in K}
        if prior_fn: pri = prior_fn(season, idx, pri)
        for wk, gw in gs.groupby("wk", sort=True):
            hist = pd.concat([prev, gs[gs.wk < wk]])
            wt = np.where(hist.season == season, 1.0, prm["w_prev"])
            if prm["hl_days"]: wt = wt * 0.5 ** ((wk - hist.date).dt.days.values / prm["hl_days"])
            fit = {}
            for c in K:
                if c == "pace":
                    oi = np.concatenate([hist.hi, hist.ai]); di = np.concatenate([hist.ai, hist.hi]); hs = np.zeros(2 * len(hist))
                    y = np.concatenate([hist.lpace, hist.lpace]) ; w = np.concatenate([wt, wt]) * 50.0
                    lam = prm["lam_pace"] * 50.0
                else:
                    oi = np.concatenate([hist.hi, hist.ai]); di = np.concatenate([hist.ai, hist.hi]); hs = np.concatenate([hist.hs, -hist.hs])
                    y = np.concatenate([hist["l_" + c], hist["l_" + c + "_a"]])
                    n = np.concatenate([hist["n_" + c], hist["n_" + c + "_a"]]); w = np.concatenate([wt, wt]) * n * lg[c] * (1 - lg[c])
                    lam = prm["lam"] * np.mean(n) * lg[c] * (1 - lg[c])
                ok = np.isfinite(y) & (w > 0)
                if ok.sum() < 30:
                    fit[c] = (pri[c][0], pri[c][1], np.average(y[ok], weights=w[ok]) if ok.any() else 0.0, 0.0); continue
                m0 = np.average(y[ok], weights=w[ok])
                O, D, h, md = solve(oi[ok], di[ok], hs[ok], y[ok] - m0, w[ok], nT, np.concatenate(pri[c]), lam)
                fit[c] = (O, D, m0 + md, h)
            if wk == gs.wk.max():
                for c in K: last[c] = (fit[c][0], fit[c][1])
            hi, ai, hs = gw.hi.values, gw.ai.values, gw.hs.values
            rh = {c: E.expit(fit[c][2] + fit[c][0][hi] + fit[c][1][ai] + hs * fit[c][3]) for c in E.COMP}
            ra = {c: E.expit(fit[c][2] + fit[c][0][ai] + fit[c][1][hi] - hs * fit[c][3]) for c in E.COMP}
            P = fit["pace"]; poss = np.exp(P[2] + P[0][hi] + P[0][ai])   # pace: offense rating of both sides (sym)
            o = pd.DataFrame({"game_id": gw.game_id.values, "season": season, "date": gw.date.values, "home": gw.team.values, "away": gw.team_a.values,
                              "neutral": gw.neutral.values, "poss_hat": poss, "act_h": gw.pts.values, "act_a": gw.pts_a.values})
            o["pred_h"] = poss * E.ppp(rh); o["pred_a"] = poss * E.ppp(ra)
            rows.append(o)
    return pd.concat(rows, ignore_index=True)

if __name__ == "__main__":
    sport = sys.argv[1]; prm = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}; tag = sys.argv[3] if len(sys.argv) > 3 else "ridge0"
    tg = pd.read_parquet(f"{HERE}/data/teamgames_{sport}.parquet")
    t0 = time.time(); p = run(tg, prm); print("secs", round(time.time() - t0, 1)); p.to_parquet(f"{HERE}/data/{tag}_{sport}.parquet")
    f = pd.read_parquet(f"{HERE}/data/frame1_{sport}.parquet")
    start = 2017 if sport == "cbb" else 2018
    common = set(p.game_id) & set(f.game_id)
    a, b = f[f.game_id.isin(common) & (f.season >= start)], p[p.game_id.isin(common) & (p.season >= start)]
    print(pd.DataFrame({"kalman frame1": H.score(a), tag: H.score(b)}).loc[["n", "team_mae", "team_bias", "margin_mae", "margin_rmse", "total_mae"]].T.to_string())
    for l in ("team_abs", "margin_sq", "total_abs"): print(H.compare(a, b, l))
