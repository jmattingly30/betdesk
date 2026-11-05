"""Round 5: more CFBD inputs on top of v4 + coach win% (margins) and v3 (totals)."""
import json, sys
exec(open('cfb_cfbd.py').read().split('def v4(')[0])
X = json.load(open(f"{C}/cfbd_extra2.json"))
def tmap(rows, key="team"): return [dict(r, tid=n2i.get(r.get(key))) for r in rows]
G = S.assign(season=S.season.astype(int))
ngames = pd.concat([G[["season", "home_id"]].set_axis(["season", "tid"], axis=1), G[["season", "away_id"]].set_axis(["season", "tid"], axis=1)]).value_counts()
SP = {(r["year"], r["tid"]): (r["rating"], r["offense"]["rating"], r["defense"]["rating"]) for r in tmap(X["sp"]) if r["tid"] is not None}
FPI = {(r["year"], r["tid"]): r["fpi"] for r in tmap(X["fpi"]) if r["tid"] is not None}
RET = {(r["season"], r["tid"]): (r["percentPPA"], r["percentPassingPPA"], r["usage"]) for r in tmap(X["returning"]) if r["tid"] is not None}
PACE = {}
for r in tmap(X["adv"]):
    g = ngames.get((r["season"], r["tid"]))
    if r["tid"] is not None and g: PACE[(r["season"], r["tid"])] = (r["offense"]["plays"] + r["defense"]["plays"]) / g
pin, pout = {}, {}
for r in X["portal"]:
    st = r.get("stars") or 2
    for k, dct in (("destination", pin), ("origin", pout)):
        t = n2i.get(r.get(k))
        if t is not None: dct[(r["season"], t)] = dct.get((r["season"], t), 0) + max(st - 2, 0)
REC = {int(r["athleteId"]): r["rating"] for r in X["recruits"] if r.get("athleteId") and r.get("rating")}
qf = pd.read_parquet("cfb_qb_first.parquet").dropna(); qf["rr"] = qf.passer_player_id.astype(int).map(REC)
QR = qf.set_index(["game_id", "pos_team_id"]).rr
# coach: SP+ change at his schools in prior seasons (program builder), and the pace of the team he led last season
coT = co.set_index(["season", "tid"]).coach; byc = co.groupby("coach")
def coach_feats(s, t):
    c = coT.get((s, t))
    if c is None: return 0.0, PACE.get((s - 1, t), np.nan)
    hist = byc.get_group(c); hist = hist[hist.season < s]
    ch = [SP[(a, b)][0] - SP[(a - 1, b)][0] for a, b in zip(hist.season, hist.tid) if (a, b) in SP and (a - 1, b) in SP]
    build = np.sum(ch) / (len(ch) + 3) if ch else 0.0
    prev = hist[hist.season == s - 1]
    pace = PACE.get((s - 1, int(prev.tid.iloc[0])), np.nan) if len(prev) else PACE.get((s - 1, t), np.nan)
    return build, pace
def add(d):
    s = d.season.values.astype(int)
    for side, col in (("h", "home_id"), ("a", "away_id")):
        t = d[col].values.astype(int)
        d["sp_" + side] = [SP.get((a - 1, b), (np.nan,) * 3)[0] for a, b in zip(s, t)]
        d["spo_" + side] = [SP.get((a - 1, b), (np.nan,) * 3)[1] for a, b in zip(s, t)]
        d["spd_" + side] = [SP.get((a - 1, b), (np.nan,) * 3)[2] for a, b in zip(s, t)]
        d["fpi_" + side] = [FPI.get((a - 1, b), np.nan) for a, b in zip(s, t)]
        d["rp_" + side] = [RET.get((a, b), (np.nan,) * 3)[0] for a, b in zip(s, t)]
        d["rpp_" + side] = [RET.get((a, b), (np.nan,) * 3)[1] for a, b in zip(s, t)]
        d["pn_" + side] = [pin.get((a, b), 0) - pout.get((a, b), 0) for a, b in zip(s, t)]
        d["pin_" + side] = [pin.get((a, b), 0) for a, b in zip(s, t)]
        d["pace_" + side] = [PACE.get((a - 1, b), np.nan) for a, b in zip(s, t)]
        cf = [coach_feats(a, b) for a, b in zip(s, t)]
        d["cb_" + side] = [x[0] for x in cf]; d["cpace_" + side] = [x[1] for x in cf]
        d["qr_" + side] = [QR.get((g, b), np.nan) for g, b in zip(d.game_id.values, t)]
    for k in ["sp", "spo", "spd", "fpi", "rp", "rpp", "pn", "pin", "pace", "cb", "cpace", "qr"]:
        for side in "ha":
            c = k + "_" + side; d["m_" + c] = d[c].isna().astype(float)
    return d
for d in (fit, te): add(d)
mu = {c: np.nanmean(np.r_[fit[c + "_h"], fit[c + "_a"]]) for c in ["sp", "spo", "spd", "fpi", "rp", "rpp", "pace", "cpace", "qr"]}
for d in (fit, te):
    for k, m in mu.items():
        for side in "ha": d[k + "_" + side] = d[k + "_" + side].fillna(m)
    for k in ["sp", "fpi", "rp", "rpp", "pn", "pin", "cb", "qr"]: d["x_" + k] = d[k + "_h"] - d[k + "_a"]
    d["x_tot_sp"] = d.spo_h + d.spo_a + d.spd_h + d.spd_a
    d["x_pace"] = d.pace_h + d.pace_a; d["x_cpace"] = d.cpace_h + d.cpace_a
    d["x_qrm"] = d.m_qr_h - d.m_qr_a
print("coverage te: sp", 1 - te.m_sp_h.mean(), "qr", 1 - te.m_qr_h.mean(), "cpace", 1 - te.m_cpace_h.mean())
def v4(e=(), l=()):
    pe = run(['model_margin'] + P + RT + D + ['c_cwin'] + list(e)); pl = run(['model_margin'] + P + ['c_cwin'] + list(l)); return np.where(w, pe, pl)
B = v4(); rep("v3", v3); rep("v4+cwin (base)", B, base=v3)
def vs(name, p, base, y='am'):
    rep(name, p, y=y, base=v3 if y == 'am' else te.model_total.values)
    d_ = np.abs(p - te[y].values) - np.abs(base - te[y].values); print(f"      vs base {d_.mean():+.4f} (t {d_.mean() / (d_.std() / np.sqrt(len(d_))):.1f})")
for nm, e, l in [("prior SP+", ['x_sp'], ['x_sp']), ("prior SP+ wk1-3", ['x_sp'], []), ("prior FPI", ['x_fpi'], ['x_fpi']), ("prior FPI wk1-3", ['x_fpi'], []),
                 ("CFBD returning PPA", ['x_rp', 'x_rpp'], []), ("portal net stars", ['x_pn'], []), ("portal in stars", ['x_pin'], []),
                 ("coach builder", ['x_cb'], ['x_cb']), ("SP+ + FPI + ret + portal wk1-3", ['x_sp', 'x_fpi', 'x_rp', 'x_rpp', 'x_pn'], [])]:
    vs("v4+ " + nm, v4(e, l), B)
print("-- known starting QB recruit rating (needs pregame starter)")
vs("v4+ known QB recruit rating", v4(['x_qr', 'x_qrm'], ['x_qr', 'x_qrm']), B)
print("-- totals (base v3 total re-fit)")
T0 = run(['model_total'], y='at'); rep("v3 total raw", te.model_total.values, y='at'); vs("v3 total stacked", T0, te.model_total.values, y='at')
for nm, c in [("prior pace", ['x_pace']), ("coach pace", ['x_cpace']), ("prior SP+ off+def", ['x_tot_sp']), ("pace+coach pace+SP", ['x_pace', 'x_cpace', 'x_tot_sp'])]:
    vs("total+ " + nm, run(['model_total'] + c, y='at'), T0, y='at')
