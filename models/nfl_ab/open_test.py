"""NFL Model B test (J 2026-11-05 rollout): the first NFL OPENER test. Until now NFL had closes only (research said our NFL
game gaps lose at the close). OddsPapi history (op/us/hist_nfl.jsonl -> lines.py -> op/us/lines_nfl.csv) has 107 2026 games
(weeks 1-7) with Pinnacle / DraftKings / FanDuel openers and closes. Model A = rebuild stack (stack_nfl.parquet, walk-forward
ratings + starting QB + weather + missing regulars; note the backtest knew the actual starter, the opener sometimes did not).
Also: Model A on J's yardstick (implied closing team totals) for 2026 and 2018-25.
python3 open_test.py -> open_test.json"""
import os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(f"{HERE}/../..")
S = pd.read_parquet(f"{ROOT}/models/rebuild/football/data/stack_nfl.parquet")
TM = json.load(open(f"{ROOT}/app/data/nfl_teams.json"))["teams"]
name2ab = {v["name"]: k for k, v in TM.items()}
L = pd.read_csv(f"{ROOT}/op/us/lines_nfl.csv")
dec2p = lambda a, b: (1 / a) / (1 / a + 1 / b)
from scipy.stats import norm
SDM, SDT = 13.0, 13.4
rows = []
for fid, g in L.groupby("fid"):
    r0 = g.iloc[0]; h, a = name2ab.get(r0.p1), name2ab.get(r0.p2)
    if h is None or a is None: continue
    rec = dict(fid=fid, start=r0.start, h=h, a=a)
    for when in ("open", "close"):
        ms, ts, secs = [], [], []
        for r in g.itertuples():
            sp, pa, pb = getattr(r, f"spread_{when}"), getattr(r, f"spread_{when}_a"), getattr(r, f"spread_{when}_b")
            if sp == sp and pa == pa and pb == pb:
                p = dec2p(pa, pb); ms.append(-sp + SDM * norm.ppf(min(max(p, .02), .98)))     # home expected margin implied
            tt, oa, ob = getattr(r, f"total_{when}"), getattr(r, f"total_{when}_a"), getattr(r, f"total_{when}_b")
            if tt == tt and oa == oa and ob == ob: ts.append(tt + SDT * norm.ppf(min(max(dec2p(oa, ob), .02), .98)))
            if when == "open": secs.append(r.spread_open_secs)
        rec[f"m_{when}"] = np.mean(ms) if ms else np.nan; rec[f"t_{when}"] = np.mean(ts) if ts else np.nan
        if when == "open": rec["open_h_before"] = np.median(secs) / 3600
        # DraftKings alone (J bets CO books; DK opens earliest)
        dk = g[g.book == "draftkings"]
        if len(dk): rec[f"dk_m_{when}"] = -dk.iloc[0][f"spread_{when}"]; rec[f"dk_t_{when}"] = dk.iloc[0][f"total_{when}"]
    rows.append(rec)
O = pd.DataFrame(rows)
S26 = S[S.season == 2026].copy()
O = O.merge(S26, left_on=["h", "a"], right_on=["home", "away"], how="inner")
O["ours_m"], O["ours_t"] = O.pred_h - O.pred_a, O.pred_h + O.pred_a
O["act_m"], O["act_t"] = O.act_h - O.act_a, O.act_h + O.act_a
out = {"n_games": len(O), "median_open_hours_before": round(float(O.open_h_before.median()), 1)}
def study(kind, when, col_ours, col_act, mk):
    d = O[O[mk].notna()].copy(); d["gap"] = d[col_ours] - d[mk]; d["res"] = d[col_act] - d[mk]
    b = np.polyfit(d.gap, d.res, 1)[0]; se = np.std(d.res - b * d.gap) / np.sqrt(((d.gap - d.gap.mean()) ** 2).sum())
    o = {"n": len(d), "b_share_real": round(float(b), 3), "t": round(float(b / se), 2), "by_gap": {}}
    if when == "open":
        cl = mk.replace("open", "close"); d["mv"] = np.sign(d.gap) * (d[cl] - d[mk])
    for k in (1, 2, 3, 4):
        z = d[d.gap.abs() >= k]; w = (np.sign(z.gap) * z.res > 0).sum(); l = (np.sign(z.gap) * z.res < 0).sum()
        o["by_gap"][f"{k}+"] = {"n": int(w + l), "win": round(w / max(w + l, 1), 3), **({"clv_pts": round(float(z.mv.mean()), 2), "moved_our_way": round(float((z.mv > 0).mean()), 3)} if when == "open" else {})}
    return o
for kind, co, ca in (("spread", "ours_m", "act_m"), ("total", "ours_t", "act_t")):
    for when in ("open", "close"):
        out[f"{kind}_{when}_consensus"] = study(kind, when, co, ca, f"{'m' if kind == 'spread' else 't'}_{when}")
        out[f"{kind}_{when}_dk"] = study(kind, when, co, ca, f"dk_{'m' if kind == 'spread' else 't'}_{when}")
# Model A yardstick: implied team totals from the market (total/2 +- margin/2) vs ours, team scores vs actual
def team_err(d, mm, mt):
    ih, ia = (d[mt] + d[mm]) / 2, (d[mt] - d[mm]) / 2
    ours = np.concatenate([d.pred_h, d.pred_a]); line = np.concatenate([ih, ia]); act = np.concatenate([d.act_h, d.act_a])
    ok = ~np.isnan(line)
    return {"n_team_scores": int(ok.sum()), "A_MAE": round(float(np.abs(ours - act)[ok].mean()), 3), "line_MAE": round(float(np.abs(line - act)[ok].mean()), 3),
            "A_APE": round(float((np.abs(ours - act) / np.maximum(act, 3))[ok].mean() * 100), 1), "line_APE": round(float((np.abs(line - act) / np.maximum(act, 3))[ok].mean() * 100), 1),
            "A_closer_than_line": round(float((np.abs(ours - act) < np.abs(line - act))[ok].mean()), 3), "A_minus_line_mean": round(float((ours - line)[ok].mean()), 2),
            "A_vs_line_MAE": round(float(np.abs(ours - line)[ok].mean()), 2)}
out["A_yardstick_2026_close_consensus"] = team_err(O, "m_close", "t_close")
out["A_yardstick_2026_open_consensus"] = team_err(O, "m_open", "t_open")
H = S[(S.season >= 2018) & S.line_margin.notna() & S.line_total.notna()]
out["A_yardstick_2018_2026_nflverse_close"] = team_err(H, "line_margin", "line_total")
json.dump(out, open(f"{HERE}/open_test.json", "w"), indent=1)
print(json.dumps(out, indent=1))
