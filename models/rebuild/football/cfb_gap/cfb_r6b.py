"""Stadium, travel, crowd and bowl-chase terms."""
exec(open('cfb_r5b.py').read().split('\nT0 = run(')[0])
for d in (fit, te): ins(d, 3)
V = pd.DataFrame(json.load(open(f"{C}/cfbd_venues.json"))).set_index("id")
SS = S.copy(); SS["dt"] = pd.to_datetime(SS.start_date, utc=True)
SS = SS.merge(V[["latitude", "longitude", "elevation", "dome", "grass", "capacity"]], left_on="venue_id", right_index=True, how="left")
# each team's home base = its most used home venue that season (non-neutral)
hb = SS[(SS.neutral_site != True) & SS.venue_id.notna()].groupby(["season", "home_id"]).venue_id.agg(lambda x: x.mode().iloc[0])
att = SS[SS.neutral_site != True].groupby(["season", "home_id"]).attendance.mean()
G2 = SS.set_index("game_id")
def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, (a1, o1, a2, o2)); h = np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2
    return 3959 * 2 * np.arcsin(np.sqrt(h))
# results so far for bowl chase
res = pd.concat([SS[["season", "dt", "game_id", "home_id", "home_points", "away_points"]].set_axis(["season", "dt", "game_id", "t", "pf", "pa"], axis=1),
                 SS[["season", "dt", "game_id", "away_id", "away_points", "home_points"]].set_axis(["season", "dt", "game_id", "t", "pf", "pa"], axis=1)]).sort_values("dt")
res["win"] = (res.pf > res.pa).astype(float); res["loss"] = (res.pf < res.pa).astype(float)
g = res.groupby(["season", "t"]); res["W"] = g.win.cumsum() - res.win; res["Lo"] = g.loss.cumsum() - res.loss
RW = res.set_index(["game_id", "t"])
def add6(d):
    gi = G2.reindex(d.game_id.values)
    lat, lon, el = gi.latitude.values, gi.longitude.values, gi.elevation.astype(float).values
    d["v_el"] = np.nan_to_num(el, nan=np.nanmedian(el)) / 1000; d["v_hi"] = (d.v_el > 1.2).astype(float)
    d["v_dome"] = gi.dome.fillna(False).astype(float).values; d["v_grass"] = gi.grass.fillna(True).astype(float).values
    doy = (gi.dt.dt.dayofyear.values - 244).clip(0, 120)   # days since Sep 1
    d["v_cold"] = (1 - d.v_dome) * np.clip(np.nan_to_num(lat, nan=35) - 33, 0, None) * doy / 100
    s = d.season.values.astype(int)
    for side, col in (("h", "home_id"), ("a", "away_id")):
        t = d[col].values.astype(int)
        vb = [hb.get((a, b)) for a, b in zip(s, t)]
        d["blat_" + side] = [V.latitude.get(x, np.nan) if x is not None else np.nan for x in vb]
        d["blon_" + side] = [V.longitude.get(x, np.nan) if x is not None else np.nan for x in vb]
        d["bel_" + side] = [float(V.elevation.get(x) or np.nan) if x is not None else np.nan for x in vb]
        r = RW.reindex(list(zip(d.game_id.values, t)))
        d["W_" + side] = r.W.fillna(0).values; d["L_" + side] = r.Lo.fillna(0).values
        d["att_" + side] = [att.get((a - 1, b), np.nan) for a, b in zip(s, t)]
    for side in "ha":
        d["trv_" + side] = np.nan_to_num(hav(d["blat_" + side], d["blon_" + side], np.nan_to_num(lat, nan=35), np.nan_to_num(lon, nan=-90)), nan=0) / 1000
        d["tz_" + side] = np.nan_to_num(np.abs(d["blon_" + side] - np.nan_to_num(lon, nan=-90)) / 15, nan=0)
        d["chase_" + side] = ((d["W_" + side] == 5) & (d.wk >= 9)).astype(float)
        d["dead_" + side] = ((d["L_" + side] >= 7) & (d.wk >= 9)).astype(float)
    d["x_trv"] = d.trv_a - d.trv_h; d["x_tz"] = d.tz_a - d.tz_h
    d["x_alt"] = d.v_el * (1 - d.neu) * np.clip(1.5 - np.nan_to_num(d.bel_a / 1000, nan=0.5), 0, None) * d.v_hi
    am = np.nanmean(d.att_h); d["x_att"] = (1 - d.neu) * (np.log(d.att_h.fillna(am).clip(lower=3000)) - np.log(am))
    d["x_chase"] = d.chase_h - d.chase_a; d["x_dead"] = d.dead_h - d.dead_a
    d["x_east_west"] = (d.blon_a - d.blon_h).fillna(0) / 15 * (gi.dt.dt.hour.values <= 18)  # west team travelling east for early kick
for d in (fit, te): add6(d)
print("venue match te", te.v_el.notna().mean(), "travel nonzero", (te.trv_a > 0).mean())
E4 = ['model_margin'] + P + RT + D + ['c_cwin']; L4 = ['model_margin'] + P + ['c_cwin']
def v4(e=()): return np.where(w, run(E4 + list(e), lam=1e-4), run(L4 + list(e), lam=1e-4))
B = v4(); rep("v4+cwin", B, base=v3)
def vs(name, p, base, y='am', b0=None):
    rep(name, p, y=y, base=v3 if y == 'am' else te.model_total.values)
    d_ = np.abs(p - te[y].values) - np.abs(base - te[y].values); print(f"      vs base {d_.mean():+.4f} (t {d_.mean() / (d_.std() / np.sqrt(len(d_))):.1f})")
for nm, e in [("travel miles", ['x_trv']), ("time zones", ['x_tz']), ("west-team early kick", ['x_east_west']), ("altitude", ['x_alt']), ("crowd size", ['x_att']),
              ("bowl chase", ['x_chase']), ("eliminated", ['x_dead']), ("chase+dead", ['x_chase', 'x_dead']), ("all", ['x_trv', 'x_tz', 'x_alt', 'x_att', 'x_chase', 'x_dead'])]:
    vs("v4+ " + nm, v4(e), B)
T1 = run(['model_total', 'x_ip'], y='at')
for nm, c in [("elevation", ['v_el']), ("dome", ['v_dome']), ("grass", ['v_grass']), ("cold proxy", ['v_cold']), ("all stadium", ['v_el', 'v_dome', 'v_grass', 'v_cold'])]:
    vs("tempo total+ " + nm, run(['model_total', 'x_ip'] + c, y='at'), T1, y='at')
te.to_parquet(os.environ.get("TE", "cfb_te_all.parquet").replace(".parquet", "_r6.parquet"))
