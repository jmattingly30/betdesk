"""Per team-game unit stats (non-garbage scrimmage plays) and walk-forward opponent-adjusted ratings by week."""
import pandas as pd, numpy as np, scipy.sparse as sp_, pickle
from scipy.sparse.linalg import lsqr
rows = []
for y in range(2014, 2027):
    d = pd.read_parquet(f"cfb/espn_{y}.parquet", columns=["game_id", "pos_team_id", "def_pos_team_id", "EPA", "rush", "pass", "scrimmage_play", "wp_before", "statYardage", "EPA_success"])
    d = d[(d.scrimmage_play == True) & d.wp_before.between(0.1, 0.9) & d.EPA.notna() & d.pos_team_id.notna()]
    d["rush"] = d.rush.astype(float); d["pass"] = d["pass"].astype(float)
    d["expl"] = ((d.rush == 1) & (d.statYardage >= 12) | (d["pass"] == 1) & (d.statYardage >= 16)).astype(float)
    g = d.groupby(["game_id", "pos_team_id", "def_pos_team_id"])
    t = pd.DataFrame({"n": g.size(), "epa": g.EPA.mean(), "sr": d.assign(es=d.EPA_success.astype(float)).groupby(["game_id", "pos_team_id", "def_pos_team_id"]).es.mean(), "expl": g.expl.mean(),
        "repa": d[d.rush == 1].groupby(["game_id", "pos_team_id", "def_pos_team_id"]).EPA.mean(),
        "pepa": d[d["pass"] == 1].groupby(["game_id", "pos_team_id", "def_pos_team_id"]).EPA.mean()}).reset_index()
    t["season"] = y; rows.append(t)
U = pd.concat(rows).rename(columns={"pos_team_id": "team", "def_pos_team_id": "opp"})
U = U[U.n >= 15]
wk = pd.concat([pd.read_csv(f"cfbros/s_{y}.csv", usecols=["game_id", "week", "season_type"]) for y in range(2014, 2027)]).drop_duplicates("game_id")
U = U.merge(wk, on="game_id", how="left"); U["week"] = np.where(U.season_type == "postseason", 20, U.week); U = U.dropna(subset=["week"])
MET = ["epa", "sr", "expl", "repa", "pepa"]
for m in MET: U[m] = pd.to_numeric(U[m], errors="coerce").astype(float); U[m] = U[m].fillna(U[m].mean())
K = 4.0; OUT = {}; last = {}
def solve(q, ix, nT, prior, ys):
    n = len(q); r = np.r_[np.arange(n), np.arange(n), n + np.arange(2 * nT)]
    c = np.r_[q.team.map(ix).values, nT + q.opp.map(ix).values, np.arange(2 * nT)]
    v = np.r_[np.ones(2 * n), np.full(2 * nT, np.sqrt(K))]
    A = sp_.csr_matrix((v, (r, c)), shape=(n + 2 * nT, 2 * nT))
    return {m: lsqr(A, np.r_[ys[m], np.sqrt(K) * prior[m]], atol=1e-7, btol=1e-7)[0] for m in MET}
for s in range(2014, 2027):
    q = U[U.season == s]; teams = pd.unique(pd.concat([q.team, q.opp])); ix = {t: i for i, t in enumerate(teams)}; nT = len(teams)
    prior = {m: np.r_[[last.get((m, "o", t), 0) * 0.5 for t in teams], [last.get((m, "d", t), 0) * 0.5 for t in teams]] for m in MET}
    for w in sorted(set(q.week.unique()) | set(range(1, 21))):
        past = q[q.week < w]; cur = q[q.week == w]
        if len(past):
            mu = {m: past[m].mean() for m in MET}; x = solve(past, ix, nT, prior, {m: past[m].values - mu[m] for m in MET})
        else: x = prior
        for t_, i in ix.items(): OUT[(s, int(w), t_)] = {m: (x[m][i], x[m][nT + i]) for m in MET}
    x = solve(q, ix, nT, {m: np.zeros(2 * nT) for m in MET}, {m: q[m].values - q[m].mean() for m in MET})
    last = {(m, k, t): x[m][(0 if k == "o" else nT) + ix[t]] for m in MET for k in "od" for t in teams}
    print(s, flush=True)
pickle.dump(OUT, open("cfb_units2.pkl", "wb")); print("rows", len(OUT))
