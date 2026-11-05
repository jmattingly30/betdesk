"""NBA Model B tier rules, graded out of sample on two openers:
  (1) SBR opener: fit 2018-20, test 2021-22 (the research split, model-b-confidence confidence.json NBA_open)
  (2) 2026 DraftKings opener from OddsPapi (op/us/lines_nba.csv, Jan-Oct 2026): fit on all SBR 2018-22, test 2026 (new market, new season)
real edge = gap x (b0 + sum b_k z_k), ridge 0.1, features from research/model-b-confidence-2026-11/nba.py.
Needs /tmp/f_nba_open_sbr.parquet: run research/model-b-confidence-2026-11/nba.py first.  -> model_b_test.json ('live' = coefficients for live.py)"""
import json, numpy as np, pandas as pd
M = "/mnt/project-files/sports-analytics/"; R = M + "models/rebuild/basketball/data/"
F = ["engines_same_side", "lineup_drives", "ls_abs", "we_like_dog", "gp_mid", "b2b_any", "fav_size", "model_split"]
TEAM = {'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET', 'Golden State Warriors': 'GS', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
        'New Orleans Pelicans': 'NO', 'New York Knicks': 'NY', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
        'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SA', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTAH', 'Washington Wizards': 'WSH'}
sbr = pd.read_parquet("/tmp/f_nba_open_sbr.parquet"); sbr["gp_mid"] = sbr.gp_min.between(16, 40).astype(float)
sbr["line"] = sbr.sbr_open; sbr["line_close"] = sbr.close; sbr["tline"] = sbr.sbr_total_open
# ---- 2026 DK opener frame (same features as nba.py prep)
s = pd.read_parquet(R + "stack_nba.parquet"); s = s[s.season.isin([2026, 2027])]
b = pd.read_parquet(R + "baseline_nba.parquet", columns=["game_id", "pred_h", "pred_a"]).rename(columns={"pred_h": "b_h", "pred_a": "b_a"})
lu = pd.read_parquet(R + "ctx_nba_lineup.parquet", columns=["game_id", "pred_h", "pred_a"]).rename(columns={"pred_h": "l_h", "pred_a": "l_a"})
fr = pd.read_parquet(R + "frame1_nba.parquet", columns=["game_id", "pred_h", "pred_a"]).rename(columns={"pred_h": "f_h", "pred_a": "f_a"})
G = pd.read_parquet(M + "models/nba-team/data/games.parquet", columns=["game_id", "home_rest", "away_rest", "close", "total_close"])
d = s.merge(b, on="game_id", how="left").merge(lu, on="game_id", how="left").merge(fr, on="game_id", how="left").merge(G, on="game_id", how="left")
d["dt"] = pd.to_datetime(d.date); d = d.sort_values("dt").reset_index(drop=True)
gp = {}; nh = []; na = []
for se, h, a in zip(d.season, d.home, d.away):
    nh.append(gp.get((se, h), 0)); na.append(gp.get((se, a), 0)); gp[(se, h)] = nh[-1] + 1; gp[(se, a)] = na[-1] + 1
d["gp_min"] = np.minimum(nh, na); d["gp_mid"] = d.gp_min.between(16, 40).astype(float)
d["ours"] = d.pred_h - d.pred_a; d["today"] = d.b_h - d.b_a; d["lineup_shift"] = (d.l_h - d.l_a) - (d.f_h - d.f_a); d["ls_abs"] = d.lineup_shift.abs()
d["act"] = d.act_h - d.act_a; d["ours_t"] = d.pred_h + d.pred_a; d["act_t"] = d.act_h + d.act_a
d["b2b_any"] = ((d.home_rest == 1) | (d.away_rest == 1)).astype(float)
L = pd.read_csv(M + "op/us/lines_nba.csv"); L = L[L.book == "draftkings"].copy()
L["h"], L["a"] = L.p1.map(TEAM), L.p2.map(TEAM); L["day"] = pd.to_datetime(L.start).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
d["day"] = d.dt.dt.strftime("%Y-%m-%d")
k = d.merge(L, left_on=["home", "away", "day"], right_on=["h", "a", "day"], how="inner")
k["line"] = -k.spread_open; k["line_close"] = -k.spread_close; k["tline"] = k.total_open; k["tline_close"] = k.total_close_y if "total_close_y" in k else k.total_close
corr_home = float(np.corrcoef(k.ours, k.line)[0, 1])
dk = k
def prep(z):
    z = z[z.line.notna()].copy(); z["edge"] = z.ours - z.line; z["res"] = z.act - z.line
    z["fav_size"] = z.line.abs(); z["we_like_dog"] = (np.sign(z.edge) != np.sign(z.line)).astype(float)
    te = z.today - z.line; z["engines_same_side"] = (np.sign(te) == np.sign(z.edge)).astype(float); z["model_split"] = (z.today - z.ours).abs()
    z["lineup_drives"] = ((np.sign(z.lineup_shift) == np.sign(z.edge)) & (z.ls_abs >= 1)).astype(float)
    z["clv"] = np.sign(z.edge) * (z.line_close - z.line)
    return z
sbr = prep(sbr); dk = prep(dk)
def fit(Fr, lam=0.1):
    mu = {f: float(Fr[f].astype(float).mean()) for f in F}; sd = {f: float(Fr[f].astype(float).std() or 1) for f in F}
    def X(z): return np.column_stack([z.edge.values] + [z.edge.values * ((z[f].astype(float).values - mu[f]) / sd[f]) for f in F])
    A = np.nan_to_num(X(Fr)); y = Fr.res.values; P = np.eye(A.shape[1]) * lam * np.mean(np.diag(A.T @ A)); P[0, 0] = 0
    return np.linalg.solve(A.T @ A + P, A.T @ y), X, mu, sd
def ats(z, side):
    s_ = np.sign(side); r = np.sign(z.res); m = (r != 0) & (s_ != 0); n = int(m.sum())
    if not n: return dict(n=0)
    w = float((s_[m] == r[m]).mean())
    return dict(n=n, win=round(w, 4), roi=round(w * 100 / 110 - (1 - w), 4), se=round(float(np.sqrt(w * (1 - w) / n)), 4), clv=round(float((np.sign(side) * np.sign(z.edge) * z.clv)[m].mean()), 2))
out = {"dk2026_games": len(dk), "dk2026_corr_ours_vs_line": round(corr_home, 3)}
def rules(T, q90):
    same = np.sign(T.real) == np.sign(T.edge); g = T.edge.abs(); early = T.gp_min <= 5
    R = {}
    R["R1_conf_top10"] = np.where((T.real.abs() >= q90) & same, "A", np.where(g >= 3, "B", "C"))
    R["R2_conf_top10_or_gap5"] = np.where(((T.real.abs() >= q90) & same) | (g >= 5), "A", np.where(g >= 3, "B", "C"))
    R["R3_R2_no_early"] = np.where((((T.real.abs() >= q90) & same) | (g >= 5)) & ~early, "A", np.where(g >= 3, "B", "C"))
    R["R4_R1_no_early"] = np.where((T.real.abs() >= q90) & same & ~early, "A", np.where(g >= 3, "B", "C"))
    return {k: {t: ats(T[v == t], T.edge[v == t]) for t in "ABC"} for k, v in R.items()}
for tag, Fr, T in (("sbr_fit18_20_test21_22", sbr[sbr.season <= 2020], sbr[sbr.season >= 2021].copy()), ("sbr_all_to_dk2026", sbr, dk.copy())):
    bb, X, mu, sd = fit(Fr); Fr = Fr.copy(); Fr["real"] = np.nan_to_num(X(Fr)) @ bb; q90 = float(Fr.real.abs().quantile(0.9))
    T["real"] = np.nan_to_num(X(T)) @ bb
    out[tag] = {"q90": round(q90, 3), "n": len(T), "rules": rules(T, q90), "raw_gap": {f"{g}+": ats(T[T.edge.abs() >= g], T.edge[T.edge.abs() >= g]) for g in (2, 3, 4, 5, 6)},
                "early_gp0_5_gap3": ats(T[(T.gp_min <= 5) & (T.edge.abs() >= 3)], T.edge[(T.gp_min <= 5) & (T.edge.abs() >= 3)])}
    print("==", tag, "q90", round(q90, 2)); [print(" ", k, v) for k, v in out[tag]["rules"].items()]; print("  raw", out[tag]["raw_gap"]); print("  early", out[tag]["early_gp0_5_gap3"])
# ---- totals at the opener: raw gap and slate-median guard (gap minus that day's median gap)
def tot(z, tl, tc):
    z = z[z[tl].notna()].copy(); z["edge"] = z.ours_t - z[tl]; z["res"] = z.act_t - z[tl]; z["clv"] = np.sign(z.edge) * (z[tc] - z[tl]) if tc in z else 0
    z["day_med"] = z.groupby(z.dt.dt.strftime("%Y-%m-%d")).edge.transform(lambda e: e.median() if len(e) >= 4 else 0.0); z["g2"] = z.edge - z.day_med
    o = {"raw": {f"{g}+": ats(z[z.edge.abs() >= g], z.edge[z.edge.abs() >= g]) for g in (3, 4, 5, 6, 8)}}
    zz = z.assign(edge=z.g2); o["slate_guard"] = {f"{g}+": ats(zz[zz.edge.abs() >= g], zz.edge[zz.edge.abs() >= g]) for g in (3, 4, 5, 6, 8)}
    o["mean_gap"] = round(float(z.edge.mean()), 2); return o
sbr["dt"] = pd.to_datetime(sbr.date)
out["totals_sbr_open_2021_22"] = tot(sbr[sbr.season >= 2021], "tline", "total_close")
out["totals_sbr_open_2018_22"] = tot(sbr, "tline", "total_close")
dk["tc"] = dk["total_close_x"] if "total_close_x" in dk else np.nan
out["totals_dk_open_2026"] = tot(dk.assign(tline=dk.total_open, tclose=dk.total_close_y if "total_close_y" in dk else dk.total_close), "tline", "tclose")
for k_ in ("totals_sbr_open_2021_22", "totals_dk_open_2026"): print(k_, json.dumps(out[k_]))
# ---- live coefficients: every opener on file (SBR 2018-22 + DK 2026)
allf = pd.concat([sbr[F + ["edge", "res"]], dk[F + ["edge", "res"]]], ignore_index=True)
bb, X, mu, sd = fit(allf); allf["real"] = np.nan_to_num(X(allf)) @ bb
out["live"] = dict(features=F, coef=dict(zip(["edge"] + F, [round(float(v), 4) for v in bb])), mu=mu, sd=sd, q90=round(float(allf.real.abs().quantile(0.9)), 3), q67=round(float(allf.real.abs().quantile(0.67)), 3))
print("live", out["live"])
json.dump(out, open("model_b_test.json", "w"), indent=1)
