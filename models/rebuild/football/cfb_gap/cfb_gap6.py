exec(open('cfb_gap3.py').read().split('v3 = te.model_margin.values')[0])
import time; t0 = time.time()
GG = G[G.margin.notna()].copy(); GG["mc"] = GG.margin.clip(-28, 28)
teams = pd.unique(pd.concat([G.home_id, G.away_id])); ix = {t: i for i, t in enumerate(teams)}; nT = len(teams)
srm = {}; final = np.zeros(nT)
for s in sorted(G.season.unique()):
    prior = 0.6 * final; gs = G[G.season == s]
    for w in sorted(gs.wk.unique()):
        h = GG[((GG.season == s) & (GG.wk < w)) | (GG.season == s - 1)]
        wt = np.where(h.season == s, 1.0, 0.15)
        n = len(h); A = sp.lil_matrix((n + nT, nT + 1))
        hi = h.home_id.map(ix).values; ai = h.away_id.map(ix).values; sw = np.sqrt(wt)
        A[np.arange(n), hi] = sw; A[np.arange(n), ai] = -sw; A[np.arange(n), nT] = sw * (~h.neutral.values).astype(float)
        lam = np.sqrt(4.0); A[n + np.arange(nT), np.arange(nT)] = lam
        y = np.r_[sw * h.mc.values, lam * prior]
        x = lsqr(A.tocsr(), y, atol=1e-8, btol=1e-8)[0]
        for g in gs[gs.wk == w].itertuples(): srm[g.game_id] = x[ix[g.home_id]] - x[ix[g.away_id]] + (0 if g.neutral else x[nT])
    final = x[:nT]
print("secs", round(time.time() - t0))
for d in (fit, te): d["srm"] = d.game_id.map(srm).fillna(0)
v3 = te.model_margin.values; D = ['d_' + c for c in RC]
base = ['rm', 'mk'] + D + ['pk', 'kal', 'r0', 'd_ro', 'd_rp', 's1', 's23']
rep("v3", v3)
rep("score-margin rating alone", run(['srm']), base=v3)
rep("C", run(base), base=v3)
rep("C + score-margin rating", run(base + ['srm']), base=v3)
