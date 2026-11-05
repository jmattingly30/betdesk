exec(open('cfb_gap2.py').read().split('rep("v3", v3)')[0])
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
G = pd.read_parquet(f'{M}/cfb-v3/game_predictions.parquet')
srs = {}
for s, g in G[G.margin.notna()].groupby("season"):
    teams = pd.unique(pd.concat([g.home_id, g.away_id])); ix = {t: i for i, t in enumerate(teams)}; n = len(g)
    A = sp.lil_matrix((n + len(teams), len(teams) + 1))
    for k, (h, a, neu) in enumerate(zip(g.home_id, g.away_id, g.neutral)):
        A[k, ix[h]] = 1; A[k, ix[a]] = -1; A[k, len(teams)] = 0 if neu else 1
    for i in range(len(teams)): A[n + i, i] = np.sqrt(3.0)   # light shrink
    y = np.r_[np.clip(g.margin.values, -35, 35), np.zeros(len(teams))]
    x = lsqr(A.tocsr(), y)[0]
    for t, i in ix.items(): srs[(s, t)] = x[i]
def feat(df):
    for L in (1, 2, 3):
        df[f"s{L}"] = [srs.get((s - L, h), np.nan) for s, h in zip(df.season, df.home_id)]
        df[f"s{L}"] = df[f"s{L}"] - pd.Series([srs.get((s - L, a), np.nan) for s, a in zip(df.season, df.away_id)], index=df.index)
        df[f"s{L}"] = df[f"s{L}"].fillna(0)
    df["s23"] = (df.s2 + df.s3) / 2
    return df
fit = feat(fit); te = feat(te)
v3 = te.model_margin.values; D = ['d_' + c for c in RC]
base = ['rm', 'mk'] + D + ['pk', 'kal', 'r0', 'd_ro', 'd_rp']
rep("v3", v3)
rep("B = A + all", run(base), base=v3)
rep("B + program (last 3 seasons SRS)", run(base + ['s1', 's2', 's3']), base=v3)
rep("B + program s1, s23", run(base + ['s1', 's23']), base=v3)
rep("B + program, ridge", run(base + ['s1', 's2', 's3'], lam=50), base=v3)
