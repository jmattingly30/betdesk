from cfb_lib import *
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
G = pd.read_parquet('/mnt/project-files/sports-analytics/models/cfb-v3/game_predictions.parquet')
G = G[G.margin.notna()].copy(); G["hp"] = (G.total + G.margin) / 2; G["ap"] = (G.total - G.margin) / 2
off, de = {}, {}
for s, g in G.groupby("season"):
    teams = pd.unique(pd.concat([g.home_id, g.away_id])); ix = {t: i for i, t in enumerate(teams)}; n = len(g); nT = len(teams)
    A = sp.lil_matrix((2 * n + 2 * nT, 2 * nT + 1)); y = np.zeros(2 * n + 2 * nT)
    for k, (h, a, hp, ap) in enumerate(zip(g.home_id, g.away_id, g.hp, g.ap)):
        A[2 * k, ix[h]] = 1; A[2 * k, nT + ix[a]] = 1; A[2 * k, 2 * nT] = 1; y[2 * k] = min(hp, 63)
        A[2 * k + 1, ix[a]] = 1; A[2 * k + 1, nT + ix[h]] = 1; A[2 * k + 1, 2 * nT] = 1; y[2 * k + 1] = min(ap, 63)
    for i in range(2 * nT): A[2 * n + i, i] = 1.5
    x = lsqr(A.tocsr(), y)[0]
    for t, i in ix.items(): off[(s, t)] = x[i]; de[(s, t)] = x[nT + i]
QF = pd.read_parquet("cfb_qb_feat.parquet").set_index(["game_id", "team"])
for d in (fit, te):
    for L in (1, 2):
        o = lambda t: np.array([off.get((s - L, x), 0) for s, x in zip(d.season, t)]); q = lambda t: np.array([de.get((s - L, x), 0) for s, x in zip(d.season, t)])
        d[f"pt{L}"] = o(d.home_id) + o(d.away_id) + q(d.home_id) + q(d.away_id)
    d["qsum"] = QF.dq.reindex(list(zip(d.game_id, d.home_id))).fillna(0).values + QF.dq.reindex(list(zip(d.game_id, d.away_id))).fillna(0).values
v3t = te.model_total.values
rep("v3 total", v3t, y='at')
rep("v3 + program scoring (last 2 seasons)", run(['model_total', 'pt1', 'pt2'], y='at'), y='at', base=v3t)
rep("v3 + known QBs (sum)", run(['model_total', 'qsum'], y='at'), y='at', base=v3t)
rep("v3 + program + rebuild total", run(['model_total', 'pt1', 'pt2', 'rt'], y='at'), y='at', base=v3t)
