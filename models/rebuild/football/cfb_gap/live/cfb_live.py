"""This week's CFB projections for the Drive Lab: Bet Desk v3 (unblended) + the tested v4 weeks 4+ terms
(3-yr program SRS, coach career win%, crowd size) for margins and in-season tempo for totals."""
exec(open('cfb_r6b.py').read().split('\nE4 = [')[0])
import scipy.sparse as sp_
from scipy.sparse.linalg import lsqr
L4 = ['model_margin'] + P + ['c_cwin', 'x_att']
def coefs(cols, y, m):
    F = fit[m]; X = np.c_[np.ones(len(F)), F[cols].values]; return np.linalg.solve(X.T @ X + 1e-4 * np.eye(X.shape[1]), X.T @ F[y].values)
bm = coefs(L4, 'am', fit.wk > 3); bt = coefs(['model_total', 'x_ip'], 'at', fit.wk > 3)
print("margin coefs", dict(zip(['c'] + L4, bm.round(3)))); print("total coefs", dict(zip(['c', 'model_total', 'x_ip'], bt.round(3))))
# upcoming games
pr = [g for g in json.load(open("probe.json"))["games"] if g["sp"] == "CFB"]
s26 = pd.read_csv("cfbros/s_2026.csv"); s26["game_id"] = s26.game_id.astype(str)
U = pd.DataFrame(pr).merge(s26[["game_id", "week", "home_id", "away_id", "venue_id"]], left_on="id", right_on="game_id", how="left")
print("unmatched", U.home_id.isna().sum()); U = U.dropna(subset=["home_id"]).copy()
U["season"] = 2026; U["wk"] = U.week; U["home_id"] = U.home_id.astype(int); U["away_id"] = U.away_id.astype(int); U["neu"] = U.n.astype(float)
U["model_margin"] = U.hs - U["as"]; U["model_total"] = U.hs + U["as"]
# program SRS (same as cfb_gap3)
M = '/mnt/project-files/sports-analytics/models'; G = pd.read_parquet(f'{M}/cfb-v3/game_predictions.parquet'); srs = {}
for s, g in G[G.margin.notna()].groupby("season"):
    teams = pd.unique(pd.concat([g.home_id, g.away_id])); ix = {t: i for i, t in enumerate(teams)}; n = len(g)
    A = sp_.lil_matrix((n + len(teams), len(teams) + 1))
    for k, (h, a, neu) in enumerate(zip(g.home_id, g.away_id, g.neutral)): A[k, ix[h]] = 1; A[k, ix[a]] = -1; A[k, len(teams)] = 0 if neu else 1
    for i in range(len(teams)): A[n + i, i] = np.sqrt(3.0)
    x = lsqr(A.tocsr(), np.r_[np.clip(g.margin.values, -35, 35), np.zeros(len(teams))])[0]
    for t, i in ix.items(): srs[(s, t)] = x[i]
for L in (1, 2, 3): U[f"s{L}"] = [srs.get((2026 - L, h), 0) - srs.get((2026 - L, a), 0) for h, a in zip(U.home_id, U.away_id)]
U["s23"] = (U.s2 + U.s3) / 2
feats(U)   # c_cwin (coach career win% before the season)
am_ = np.nanmean(fit.att_h)
U["att_h"] = [att.get((2025, h), np.nan) for h in U.home_id]
U["x_att"] = (1 - U.neu) * (np.log(U.att_h.fillna(am_).clip(lower=3000)) - np.log(am_))
# tempo: all 2026 games so far, shrunk K=3 toward coach pace prior
PG26 = PG[PG.season == 2026].groupby("team").agg(n=("gplays", "size"), s=("gplays", "sum"))
for side, col in (("h", "home_id"), ("a", "away_id")):
    cp = np.array([coach_feats(2026, t)[1] for t in U[col]]); cp = np.where(np.isnan(cp), mu["cpace"], cp)
    n = U[col].map(PG26.n).fillna(0).values; s = U[col].map(PG26.s).fillna(0).values
    U["ip_" + side] = (s + 3 * cp) / (n + 3); U["gp_" + side] = n
U["x_ip"] = U.ip_h + U.ip_a
U["m_new"] = np.c_[np.ones(len(U)), U[L4].values] @ bm; U["t_new"] = np.c_[np.ones(len(U)), U[['model_total', 'x_ip']].values] @ bt
U["hs_new"] = (U.t_new + U.m_new) / 2; U["as_new"] = (U.t_new - U.m_new) / 2
U["plays_exp"] = U.x_ip / 2   # expected plays for both teams in this game, per-team average scale -> game plays
cols = ["id", "k", "h", "a", "n", "hs", "as", "hs_new", "as_new", "m_new", "t_new", "x_ip", "ln"]
U[cols].to_json("cfb_live.json", orient="records")
# shared file for the Bet Desk (Model A = hs/as from the v4 margin + tempo total; v3_hs/v3_as = the desk's v3 inputs)
import datetime
A = [{"id": r.id, "h": r.h, "a": r.a, "k": r.k, "n": int(r.n), "hs": round(r.hs_new, 2), "as": round(r.as_new, 2), "m_new": round(r.m_new, 2),
      "t_new": round(r.t_new, 2), "v3_hs": r.hs, "v3_as": r["as"], "wk": int(r.wk)} for _, r in U.iterrows()]
json.dump({"made": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "model": "CFB rebuild v4 (weeks 4+): v3 + program SRS + coach career win% + crowd size; total v3 + tempo",
           "games": A}, open("/mnt/project-files/sports-analytics/app/data/cfb_live.json", "w"))
print("wrote app/data/cfb_live.json", len(A))
print(U[["a", "h", "model_margin", "m_new", "model_total", "t_new", "x_ip"]].round(1).to_string())
