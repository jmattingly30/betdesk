"""Data for the Drive Lab viewer: this week's NFL games from the rebuild (shadow projections = full stack where the
inputs exist), the key-number score curve for margin and total (the kept distribution), 2,000 drive-by-drive
simulations per game (score shapes) and 12 sample games to replay. -> ../viewer/nfl_drives.json"""
import sys, os, json, datetime as dt, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from scipy import stats
import sim as S, trace as TR
hist = json.load(open(f"{HERE}/../shadow/history.json"))
games = sorted([v for v in hist.values() if v["sp"] == "NFL" and pd.Timestamp(v["k"]) > pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=4)], key=lambda v: v["k"])
p = pd.DataFrame({"pred_h": [g["hs"] for g in games], "pred_a": [g["as"] for g in games]})
KW = dict(k_score=0.1, k_late=0.6, th_sd=0.3)
q, mh = S.simulate(p, nsim=2000, **KW)
# key-number curve (keynum.py): normal x weights fit on 2010-2017
st = pd.read_parquet(f"{HERE}/data/stack_nfl.parquet"); tr = st[(st.season >= 2010) & (st.season <= 2017)]
mu_m, mu_t = (tr.pred_h - tr.pred_a).values, (tr.pred_h + tr.pred_a).values; am, at = (tr.act_h - tr.act_a).values, (tr.act_h + tr.act_a).values
SDM, SDT = np.std(am - mu_m), np.std(at - mu_t); M = np.arange(-60, 61); TT = np.arange(0, 121)
def W(mu, act, grid, sd):
    dens = stats.norm.pdf(grid[None, :], mu[:, None], sd).mean(0); freq = pd.Series(act).value_counts(normalize=True).reindex(grid, fill_value=0).values
    return np.where(dens > 2e-3, (freq + 1e-4) / (dens + 1e-4), 1.0)
wm, wt = W(mu_m, am, M, SDM), W(mu_t, at, TT, SDT)
def pmf(mu, grid, sd, w): d = stats.norm.pdf(grid, mu, sd) * w; return d / d.sum()
out = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "sport": "NFL", "nsim": 2000,
       "note": "Projections from the rebuilt NFL model (drive-component ratings, starting QB, dome; injuries and wind are not wired for future games yet). Win chances and margin/total shapes use the key-number curve, which tested better than the raw simulator; the simulator supplies the team-score shapes and the drive-by-drive replays.",
       "games": []}
for i, g in enumerate(games):
    mm, tt = g["hs"] - g["as"], g["hs"] + g["as"]
    pm, pt = pmf(mm, M, SDM, wm), pmf(tt, TT, SDT, wt)
    ph = float(pm[M > 0].sum() + 0.5 * pm[M == 0].sum())
    sims = [TR.trace_game(q.th_h.iat[i], q.th_a.iat[i], seed=s + 7919 * (i + 1), **KW) for s in range(2000)]
    Hs = np.array([s["home"] for s in sims]); As = np.array([s["away"] for s in sims])
    hist_ = lambda x: [[int(v), int(c)] for v, c in zip(*np.unique(x, return_counts=True))]
    lm = g.get("ln"); lt = g.get("lt")
    out["games"].append({"id": g["id"], "kick": g["k"], "home": g["h"], "away": g["a"],
        "proj": {"home": round(g["hs"], 1), "away": round(g["as"], 1), "margin": round(mm, 1), "total": round(tt, 1)},
        "p_home": round(ph, 3),
        "key": {k: round(float(pm[M == k].sum() + pm[M == -k].sum()), 3) for k in (3, 7, 10, 14)},
        "margin_pmf": [[int(m), round(float(v), 4)] for m, v in zip(M, pm) if abs(m) <= 35],
        "total_pmf": [[int(t), round(float(v), 4)] for t, v in zip(TT, pt) if 15 <= t <= 85],
        "q_margin": [int(M[np.searchsorted(np.cumsum(pm), z)]) for z in (0.1, 0.5, 0.9)],
        "q_total": [int(TT[np.searchsorted(np.cumsum(pt), z)]) for z in (0.1, 0.5, 0.9)],
        "hist_home": hist_(Hs), "hist_away": hist_(As),
        "drives_mean": round(float(np.mean([len(s["drives"]) for s in sims])), 1),
        "line": {"spread": lm, "total": lt}, "samples": sims[:12]})
    print(g["a"], "@", g["h"], out["games"][-1]["proj"], out["games"][-1]["p_home"], flush=True)
json.dump(out, open(f"{HERE}/../viewer/nfl_drives.json", "w"), separators=(",", ":"))
print("wrote", len(out["games"]), "games", os.path.getsize(f"{HERE}/../viewer/nfl_drives.json") // 1024, "KB")
