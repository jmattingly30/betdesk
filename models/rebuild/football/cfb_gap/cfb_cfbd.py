from cfb_lib import *
RD = "cfbros"; C = "/mnt/project-files/sports-analytics/data/cfbd"
S = pd.concat([pd.read_csv(f"{RD}/s_{y}.csv") for y in range(2014, 2027)])
n2i = pd.concat([S[["home_team", "home_id"]].set_axis(["t", "i"], axis=1), S[["away_team", "away_id"]].set_axis(["t", "i"], axis=1)]).dropna().drop_duplicates("t").set_index("t").i
tal = pd.read_csv(f"{C}/cfbd_talent.csv"); tal["tid"] = tal.team.map(n2i)
rec = pd.read_csv(f"{C}/cfbd_recruiting.csv"); rec["tid"] = rec.team.map(n2i)
co = pd.read_csv(f"{C}/cfbd_coaches.csv"); co["tid"] = co.team.map(n2i)
print("unmapped share talent/rec/coach:", tal.tid.isna().mean().round(3), rec.tid.isna().mean().round(3), co.tid.isna().mean().round(3))
TAL = tal.dropna(subset=["tid"]).groupby(["season", "tid"]).talent.mean()
R = rec.dropna(subset=["tid"]).groupby(["season", "tid"]).points.mean()
def rec4(s, t): return np.nansum([R.get((y, t), np.nan) * wt for y, wt in zip(range(s - 3, s + 1), (0.15, 0.25, 0.3, 0.3))]) if any((y, t) in R.index for y in range(s - 3, s + 1)) else np.nan
# head coach of record = most games that season
co = co.dropna(subset=["tid"]).sort_values("games").groupby(["season", "tid"]).tail(1)
HC = co.set_index(["season", "tid"]).coach
car = co.sort_values("season").copy(); car["w_cum"] = car.groupby("coach").wins.cumsum() - car.wins; car["g_cum"] = car.groupby("coach").games.cumsum() - car.games
CW = car.set_index(["season", "tid"])
def feats(d):
    def side(t):
        s = d.season.values
        ta = np.array([TAL.get((a, b), np.nan) for a, b in zip(s, t)]); r4 = np.array([rec4(a, b) for a, b in zip(s, t)])
        new = np.array([float(HC.get((a, b)) is not None and HC.get((a - 1, b)) is not None and HC.get((a, b)) != HC.get((a - 1, b))) for a, b in zip(s, t)])
        cw = np.array([((CW.w_cum.get((a, b), 0) + 5) / (CW.g_cum.get((a, b), 0) + 10)) if (a, b) in CW.index else 0.5 for a, b in zip(s, t)])
        return ta, r4, new, cw
    h = side(d.home_id.values); a = side(d.away_id.values)
    for k, nm in enumerate(["tal", "rec", "newc", "cwin"]):
        x, y = h[k], a[k]; mu = np.nanmean(np.r_[x, y])
        d["c_" + nm] = np.nan_to_num(x, nan=mu) - np.nan_to_num(y, nan=mu)
    d["c_tal"] /= 100; d["c_rec"] /= 100
    return d
for d in (fit, te): feats(d)
print("talent coverage fit (2015+ only):", np.isfinite(fit.c_tal).mean())
def v4(extra_e=(), extra_l=()):
    pe = run(['model_margin'] + P + RT + D + list(extra_e)); pl = run(['model_margin'] + P + list(extra_l)); return np.where(w, pe, pl)
V4 = v4(); rep("v3", v3); rep("v4", V4, base=v3)
for nm, e, l in [("+ talent", ['c_tal'], []), ("+ recruiting 4yr", ['c_rec'], []), ("+ new coach", ['c_newc'], ['c_newc']), ("+ coach career win%", ['c_cwin'], ['c_cwin']),
                 ("+ talent + recruiting (all weeks)", ['c_tal', 'c_rec'], ['c_tal', 'c_rec']), ("+ all four", ['c_tal', 'c_rec', 'c_newc', 'c_cwin'], ['c_tal', 'c_rec', 'c_newc', 'c_cwin'])]:
    p = v4(e, l); rep("v4 " + nm, p, base=v3)
    d_ = np.abs(p - te.am.values) - np.abs(V4 - te.am.values); print(f"      vs v4 {d_.mean():+.3f} (t {d_.mean() / (d_.std() / np.sqrt(len(d_))):.1f})")
