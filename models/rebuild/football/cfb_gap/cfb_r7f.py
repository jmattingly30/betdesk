"""Poll prior: AP + Coaches points share for week w (released before week w games)."""
exec(open('cfb_r6b.py').read().split('\nE4 = [')[0])
RK = json.load(open(f"{C}/cfbd_rankings.json")); PS = {}
for x in RK:
    if x["seasonType"] != "regular": continue
    for p in x["polls"]:
        if p["poll"] not in ("AP Top 25", "Coaches Poll"): continue
        mx = max(r["points"] or 0 for r in p["ranks"]) or 1
        for r in p["ranks"]:
            k = (x["season"], x["week"], r["teamId"]); PS.setdefault(k, []).append((r["points"] or 0) / mx)
PS = {k: np.mean(v) for k, v in PS.items()}
WK = pd.concat([pd.read_csv(f"cfbros/s_{y}.csv", usecols=["game_id", "week", "season_type"]) for y in range(2014, 2027)]).drop_duplicates("game_id").set_index("game_id")
for d in (fit, te):
    wv = d.game_id.map(WK.week).fillna(15).astype(int).values; st = d.game_id.map(WK.season_type).values
    wv = np.where(st == "postseason", 99, wv)
    def g(s, w_, t):
        if w_ == 99: return np.nan
        for ww in range(w_, 0, -1):   # latest poll at or before this week
            if any((s, ww, tt) in PS for tt in (t,)) : return PS[(s, ww, t)]
            if (s, ww, 61) in PS or (s, ww, 333) in PS: return 0.0   # poll exists that week, team unranked
        return 0.0
    ph = np.array([g(s, w_, t) for s, w_, t in zip(d.season.values.astype(int), wv, d.home_id.values.astype(int))])
    pa = np.array([g(s, w_, t) for s, w_, t in zip(d.season.values.astype(int), wv, d.away_id.values.astype(int))])
    d["x_poll"] = np.nan_to_num(ph - pa); d["x_rk_any"] = (np.nan_to_num(ph) > 0).astype(float) - (np.nan_to_num(pa) > 0).astype(float)
E4 = ['model_margin'] + P + RT + D + ['c_cwin', 'x_att']; L4 = ['model_margin'] + P + ['c_cwin', 'x_att']
def v4(e=(), l=()): return np.where(w, run(E4 + list(e), lam=1e-4), run(L4 + list(l), lam=1e-4))
B = v4(); rep("v4+cwin+crowd", B, base=v3)
def vs(name, p):
    rep(name, p, base=v3); d_ = np.abs(p - te.am.values) - np.abs(B - te.am.values); print(f"      vs base {d_.mean():+.4f} (t {d_.mean() / (d_.std() / np.sqrt(len(d_))):.1f})")
vs("+ poll wk1-3", v4(['x_poll'], [])); vs("+ poll all weeks", v4(['x_poll'], ['x_poll'])); vs("+ poll + ranked flag all", v4(['x_poll', 'x_rk_any'], ['x_poll', 'x_rk_any']))
