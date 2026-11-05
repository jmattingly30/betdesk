"""Walk-forward opponent-adjusted DRIVE ratings for CFB Model A's unit-sum term (research/cfb-sim-2026-10 ratings.py, drive
metrics only: fin = offense points minus expected from the drive's start, pts = points per drive, tdr = TD rate). The research's
'fin_sum' alone gave -0.027 team MAE on v3 (t -3.9, all 4 test seasons); the play-level metrics (pass EPA) need the play table,
which is not kept. For (season, week) a ridge on all earlier drives this season + last season at weight .35 (garbage time .35).
python3 unit_ratings.py [fresh pbp dir]  -> data/unit_ratings.parquet (season, week, team, <m>_o, <m>_d)
With a pbp dir (run_live.sh's 2026 download) the current season's drives are rebuilt from it first (data/drives_cur.parquet)."""
import sys, os, subprocess, tempfile, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.normpath(f"{HERE}/../../research/cfb-sim-2026-10")
PRIOR, LAM, GT = 0.35, 25.0, 0.35
D = pd.read_parquet(f"{R}/data/drives.parquet"); G = pd.read_parquet(f"{R}/data/games.parquet")
if len(sys.argv) > 1:   # fresh current season from play-by-play
    with tempfile.TemporaryDirectory() as T:
        subprocess.run(["python3", f"{R}/prep_drives.py", sys.argv[1], T], check=True, stdout=subprocess.DEVNULL)
        d2, g2 = pd.read_parquet(f"{T}/drives.parquet"), pd.read_parquet(f"{T}/games.parquet")
    d2[["game_id", "season", "off", "dfn", "res", "period", "t0", "t1", "off_pts", "def_pts", "n", "y0", "wp0"]].to_parquet(f"{HERE}/data/drives_cur.parquet")
    g2.to_parquet(f"{HERE}/data/games_cur.parquet")
if os.path.exists(f"{HERE}/data/drives_cur.parquet"):
    d2, g2 = pd.read_parquet(f"{HERE}/data/drives_cur.parquet"), pd.read_parquet(f"{HERE}/data/games_cur.parquet")
    s = set(g2.season); D = pd.concat([D[~D.season.isin(s)], d2], ignore_index=True); G = pd.concat([G[~G.season.isin(s)], g2], ignore_index=True)
G = G[G.ok].copy(); G["game_id"] = G.game_id.astype(str)
cg = pd.read_parquet(f"{HERE}/../team-strength/data/cfb_games.parquet")[["game_id", "neutral_site"]]; cg["game_id"] = cg.game_id.astype(str)
G = G.merge(cg, on="game_id", how="left"); G["wk"] = np.where(G.stype.astype(str).isin(["3", "3.0"]), 20, G.week.astype(float)); G["neu"] = G.neutral_site.fillna(False).astype(bool)
gk = G.set_index("game_id")[["wk", "neu", "home"]]
D = D.drop(columns=["home"], errors="ignore"); D["game_id"] = D.game_id.astype(str)
D = D[D.game_id.isin(gk.index) & (D.period <= 4) & D.y0.between(1, 99)].join(gk, on="game_id").dropna(subset=["off", "dfn"]).copy()
ref = D[D.season.between(2015, 2025)]; ep = ref.groupby((ref.y0 // 5).clip(0, 19).astype(int)).off_pts.mean()
D["fin"] = D.off_pts - (D.y0 // 5).clip(0, 19).astype(int).map(ep); D["pts"] = D.off_pts; D["tdr"] = D.res.isin(["TD"]).astype(float)
D["w"] = np.where(D.wp0.between(0.1, 0.9) | D.wp0.isna(), 1.0, GT)
D["ishome"] = ((D.off == D.home) & ~D.neu).astype(float) - ((D.dfn == D.home) & ~D.neu).astype(float)
M = ["fin", "pts", "tdr"]; teams = np.unique(np.r_[D.off, D.dfn]).astype(int); ix = {t: i for i, t in enumerate(teams)}; nT = len(teams)
D["io"] = D.off.astype(int).map(ix); D["id"] = D.dfn.astype(int).map(ix) + nT
def fit(X, wts):
    k = 2 * nT + 2; io, idd, hm = X.io.values, X.id.values, X.ishome.values; n = len(io)
    idx = [io, idd, np.full(n, 2 * nT), np.full(n, 2 * nT + 1)]; val = [np.ones(n), np.ones(n), np.ones(n), hm]
    A = np.zeros(k * k)
    for a, va in zip(idx, val):
        for b_, vb in zip(idx, val): A += np.bincount(a * k + b_, weights=wts * va * vb, minlength=k * k)
    A = A.reshape(k, k); A[np.arange(2 * nT), np.arange(2 * nT)] += LAM; A += 1e-9 * np.eye(k); out = {}
    for c in M:
        bv = np.zeros(k); y = X[c].values
        for a, va in zip(idx, val): bv += np.bincount(a, weights=wts * va * y, minlength=k)
        b = np.linalg.solve(A, bv); out[c] = (b[:nT], b[nT:2 * nT])
    return out
rows = []
for s in range(2015, int(G.season.max()) + 1):
    weeks = sorted(set(G[G.season == s].wk.dropna().astype(int))); last = max(weeks) if weeks else 0
    for wk in weeks + [last + 1]:   # one extra week = ratings after every game played so far (live)
        cur = D[(D.season == s) & (D.wk < wk)]; prev = D[D.season == s - 1]; Z = pd.concat([cur, prev])
        wt = Z.w.values * np.r_[np.ones(len(cur)), np.full(len(prev), PRIOR)]
        res = fit(Z, wt); f = pd.DataFrame({"team": teams, "season": s, "week": wk})
        for c, (o, d_) in res.items(): f[c + "_o"] = o; f[c + "_d"] = d_
        rows.append(f)
    print(s, len(weeks), flush=True)
U = pd.concat(rows, ignore_index=True); U.to_parquet(f"{HERE}/data/unit_ratings.parquet"); print(U.shape)
