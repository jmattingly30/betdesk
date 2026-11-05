exec(open("cfb_gap.py").read().split('v3 = te.model_margin.values')[0])
F = "/mnt/project-files/sports-analytics/models/rebuild/football/data"
def add(name, file):
    global m, fit, te
    x = pd.read_parquet(f"{F}/{file}")[["game_id", "pred_h", "pred_a"]]; x[name] = x.pred_h - x.pred_a
    for d in ("fit", "te"):
        globals()[d] = globals()[d].merge(x[["game_id", name]], on="game_id", how="left")
        globals()[d][name] = globals()[d][name].fillna(globals()[d]["rm"])
add("pk", "baseline_cfb_points.parquet"); add("kal", "frame1_cfb.parquet"); add("r0", "ridge0_cfb.parquet")
ret = pd.read_csv(f"{M}/team-strength/data/cfb_returning.csv")
for d in ("fit", "te"):
    df = globals()[d]
    for side, idc in (("h", "home_id"), ("a", "away_id")):
        df = df.merge(ret.rename(columns={"team_id": idc, "ret_off": f"ro_{side}", "ret_pass": f"rp_{side}"})[["season", idc, f"ro_{side}", f"rp_{side}"]], on=["season", idc], how="left")
    df["d_ro"] = df.ro_h.fillna(df.ro_h.mean()) - df.ro_a.fillna(df.ro_a.mean()); df["d_rp"] = df.rp_h.fillna(0) - df.rp_a.fillna(0)
    globals()[d] = df
v3 = te.model_margin.values; D = ['d_' + c for c in RC]
rep("v3", v3)
base = ['rm', 'mk'] + D
rep("A rebuild+market+roster", run(base), base=v3)
rep("A + points-kalman", run(base + ['pk']), base=v3)
rep("A + kalman frame", run(base + ['kal']), base=v3)
rep("A + pk + kal + r0", run(base + ['pk', 'kal', 'r0']), base=v3)
rep("A + returning", run(base + ['d_ro', 'd_rp']), base=v3)
rep("A + all", run(base + ['pk', 'kal', 'r0', 'd_ro', 'd_rp']), base=v3)
