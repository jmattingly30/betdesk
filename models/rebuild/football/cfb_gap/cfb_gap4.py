exec(open('cfb_gap3.py').read().split('v3 = te.model_margin.values')[0])
r_all = pd.read_parquet("/mnt/project-files/sports-analytics/models/rebuild/football/data/teamgames_cfb.parquet")[["game_id", "team", "date", "season"]]
r_all["date"] = pd.to_datetime(r_all.date); r_all = r_all.sort_values(["team", "date"])
r_all["rest"] = r_all.groupby(["team", "season"]).date.diff().dt.days
rest = r_all.set_index(["game_id", "team"]).rest
# team home edge: previous seasons' home margins minus SRS expectation (with league HFA), shrunk
G2 = G[G.margin.notna() & ~G.neutral].copy()
G2["res"] = G2.margin - np.array([srs.get((s, h), 0) - srs.get((s, a), 0) for s, h, a in zip(G2.season, G2.home_id, G2.away_id)])
G2["res"] -= G2.groupby("season").res.transform("mean")
hres = G2.groupby(["home_id", "season"]).res.agg(["sum", "count"])
def hfa(t, s, K=40):
    rows = [hres.loc[(t, y)] for y in range(s - 5, s) if (t, y) in hres.index]
    if not rows: return 0.0
    sm = sum(r["sum"] for r in rows); n = sum(r["count"] for r in rows); return sm / (n + K)
def feat2(df):
    rh = np.array([rest.get((g, h), np.nan) for g, h in zip(df.game_id, df.home_id)]); ra = np.array([rest.get((g, a), np.nan) for g, a in zip(df.game_id, df.away_id)])
    df["bye_h"] = (rh >= 13).astype(float); df["bye_a"] = (ra >= 13).astype(float)
    df["short_h"] = (rh <= 5).astype(float); df["short_a"] = (ra <= 5).astype(float)
    df["thfa"] = [0.0 if n else hfa(h, s) for h, s, n in zip(df.home_id, df.season, df.neutral)]
    return df
fit = feat2(fit); te = feat2(te)
v3 = te.model_margin.values; D = ['d_' + c for c in RC]
base = ['rm', 'mk'] + D + ['pk', 'kal', 'r0', 'd_ro', 'd_rp', 's1', 's23']
rep("v3", v3)
rep("C = B + program", run(base), base=v3)
rep("C + rest", run(base + ['bye_h', 'bye_a', 'short_h', 'short_a']), base=v3)
rep("C + team home edge", run(base + ['thfa']), base=v3)
rep("C + rest + home edge", run(base + ['bye_h', 'bye_a', 'short_h', 'short_a', 'thfa']), base=v3)
# robust fit (Huber via IRLS) on C
def run_huber(cols, delta=10.0, it=8):
    pred = np.zeros(len(te))
    for fm, tm in [(fit.wk <= 3, te.wk <= 3), (fit.wk > 3, te.wk > 3)]:
        X = np.c_[np.ones(fm.sum()), fit.loc[fm, cols].values]; y = fit.am[fm].values; w = np.ones(len(y))
        for _ in range(it):
            b = np.linalg.solve((X * w[:, None]).T @ X, (X * w[:, None]).T @ y); r = np.abs(y - X @ b); w = np.where(r <= delta, 1.0, delta / r)
        pred[tm.values] = np.c_[np.ones(tm.sum()), te.loc[tm, cols].values] @ b
    return pred
rep("C, Huber fit", run_huber(base), base=v3)
rep("C + home edge, Huber", run_huber(base + ['thfa']), base=v3)
np.save("cfb_best_pred.npy", run_huber(base + ['thfa'])); te[["game_id", "season", "wk", "am", "model_margin", "line_close", "line_open"]].to_parquet("cfb_te.parquet")
