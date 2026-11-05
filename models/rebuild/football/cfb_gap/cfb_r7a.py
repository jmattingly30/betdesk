"""Opponent-adjusted tempo: walk-forward by week, offense plays = mu + off(team) + def(opponent), ridge toward half of
last season's final values. Feature x_oip = expected plays in this matchup (both offenses)."""
exec(open('cfb_r5b.py').read().split('\nT0 = run(')[0])
import scipy.sparse as sp_
from scipy.sparse.linalg import lsqr
wk = pd.concat([pd.read_csv(f"cfbros/s_{y}.csv", usecols=["game_id", "week", "season_type"]) for y in range(2014, 2027)]).drop_duplicates("game_id")
Q = PG.merge(wk, on="game_id", how="left"); Q["week"] = np.where(Q.season_type == "postseason", 20, Q.week)
opp = Q[["game_id", "team"]].rename(columns={"team": "opp"}); Q = Q.merge(opp, on="game_id"); Q = Q[Q.team != Q.opp]
OFF = {}   # (game_id, team) -> (off_t, def_opp, mu) pre-game
K = 3.0
last = {}
for s in range(2014, 2027):
    q = Q[Q.season == s]; teams = pd.unique(pd.concat([q.team, q.opp])); ix = {t: i for i, t in enumerate(teams)}; nT = len(teams)
    prior = np.array([[last.get(("o", t), 0) * 0.5, last.get(("d", t), 0) * 0.5] for t in teams])
    for w in sorted(q.week.unique()):
        past = q[q.week < w]; cur = q[q.week == w]
        mu = past.oplays.mean() if len(past) else Q[Q.season == s - 1].oplays.mean() if s > 2014 else q.oplays.mean()
        if len(past):
            n = len(past); A = sp_.lil_matrix((n + 2 * nT, 2 * nT))
            for k, (t, o) in enumerate(zip(past.team, past.opp)): A[k, ix[t]] = 1; A[k, nT + ix[o]] = 1
            for i in range(nT): A[n + i, i] = np.sqrt(K); A[n + nT + i, nT + i] = np.sqrt(K)
            y = np.r_[past.oplays.values - mu, np.sqrt(K) * prior[:, 0], np.sqrt(K) * prior[:, 1]]
            x = lsqr(A.tocsr(), y, atol=1e-6, btol=1e-6)[0]
        else: x = np.r_[prior[:, 0], prior[:, 1]]
        for g, t, o in zip(cur.game_id, cur.team, cur.opp): OFF[(g, t)] = (x[ix[t]], x[nT + ix[o]], mu)
    # season-end values for next year's prior
    n = len(q); A = sp_.lil_matrix((n + 2 * nT, 2 * nT))
    for k, (t, o) in enumerate(zip(q.team, q.opp)): A[k, ix[t]] = 1; A[k, nT + ix[o]] = 1
    for i in range(nT): A[n + i, i] = 1; A[n + nT + i, nT + i] = 1
    x = lsqr(A.tocsr(), np.r_[q.oplays.values - q.oplays.mean(), np.zeros(2 * nT)])[0]
    last = {**{("o", t): x[ix[t]] for t in teams}, **{("d", t): x[nT + ix[t]] for t in teams}}
    print(s, "done", flush=True)
import pickle; pickle.dump(OFF, open("cfb_oip.pkl", "wb"))
for d in (fit, te):
    e = []
    for g, h, a in zip(d.game_id.values, d.home_id.values.astype(int), d.away_id.values.astype(int)):
        x1 = OFF.get((g, h)); x2 = OFF.get((g, a))
        e.append(np.nan if x1 is None or x2 is None else 2 * x1[2] + x1[0] + x1[1] + x2[0] + x2[1])
    d["x_oip"] = e; d["x_oip"] = d.x_oip.fillna(np.nanmean(d.x_oip))
print("coverage", te.x_oip.notna().mean())
v3t = te.model_total.values; T1 = run(['model_total', 'x_ip'], y='at')
def vs(name, p, base=T1):
    rep(name, p, y='at', base=v3t); d_ = np.abs(p - te["at"].values) - np.abs(base - te["at"].values); print(f"      vs tempo total {d_.mean():+.4f} (t {d_.mean() / (d_.std() / np.sqrt(len(d_))):.1f})")
vs("tempo total (raw plays)", T1)
vs("opp-adjusted tempo instead", run(['model_total', 'x_oip'], y='at'))
vs("both", run(['model_total', 'x_ip', 'x_oip'], y='at'))
