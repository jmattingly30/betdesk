"""Grade CFB Model A + Model B from history/runs.jsonl against finals (cfbfastR schedule csv, refreshed by run_live.sh).
Model A (J yardstick): team-score miss vs actual AND vs the market's implied team totals (last market seen before kickoff); v4 alone
(no usage / absence inputs) graded beside it, and games with a missing-star input split out. Model B: per tier and kind, the FIRST time
a pick reached tier A/B -> win rate, ROI at the shown price, CLV in points vs the last market seen.
python3 models/cfb_ab/grade.py [schedule csv] -> models/cfb_ab/history/grade.json"""
import json, os, sys, datetime as dt, collections, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); H = f"{HERE}/history"
runs = [json.loads(l) for l in open(f"{H}/runs.jsonl")] if os.path.exists(f"{H}/runs.jsonl") else []
src = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/../rebuild/football/cfb_gap/live/cfbros/s_2026.csv"
S = pd.read_csv(src, usecols=["game_id", "completed", "home_points", "away_points"]); S = S[S.completed == True].dropna(subset=["home_points"])
fin = {str(r.game_id): {"h": float(r.home_points), "a": float(r.away_points)} for r in S.itertuples()}
first, last = {}, {}
for r in runs:
    if r.get("k") and r["t"][:16] > r["k"][:16].replace("T", " "): continue   # only pre-kickoff runs
    last[r["id"]] = r
    for k, z in r["B"].items():
        if z["tier"] in "AB" and z.get("bet") and (r["id"], k) not in first: first[(r["id"], k)] = (r, z)
out = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "A": {}, "B": {}}
E = collections.defaultdict(list)
for i, f in fin.items():
    r = last.get(i)
    if not r: continue
    sp, tt = r["B"].get("spread"), r["B"].get("total"); hasabs = bool(r.get("abs"))
    imp = None
    if sp and tt and sp.get("market") is not None and tt.get("market") is not None: imp = ((tt["market"] + sp["market"]) / 2, (tt["market"] - sp["market"]) / 2)
    for j, side in ((0, "h"), (1, "a")):
        E["A"].append(abs(r["A"][j] - f[side])); E["v4"].append(abs(r["v4"][j] - f[side]))
        if hasabs: E["A_abs"].append(abs(r["A"][j] - f[side])); E["v4_abs"].append(abs(r["v4"][j] - f[side]))
        if imp: E["line"].append(abs(imp[j] - f[side])); E["closer"].append(abs(r["A"][j] - f[side]) < abs(imp[j] - f[side]))
for k, v in E.items():
    if v: out["A"][k] = {"n": len(v), ("share" if k == "closer" else "MAE"): round(sum(v) / len(v), 3)}
T = collections.defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0, "clv": 0.0})
for (gid, k), (r, z) in first.items():
    f = fin.get(gid)
    if not f: continue
    b = z["bet"]; pt = b["pt"]; d = 1 + (b["price"] / 100 if b["price"] > 0 else 100 / -b["price"])
    lk = last[gid]["B"].get(k if k != "ml" else "spread", {}).get("market")
    if k == "spread":
        m = f["h"] - f["a"]; res = (m + pt) if z["side"] == r["h"] else (-m + pt); clv = None if lk is None else ((lk - z["market"]) if z["side"] == r["h"] else (z["market"] - lk))
    elif k == "total":
        t = f["h"] + f["a"]; res = (t - pt) if z["side"] == "Over" else (pt - t); clv = None if lk is None else ((lk - z["market"]) if z["side"] == "Over" else (z["market"] - lk))
    else: res = (f["h"] - f["a"]) if z["side"] == r["h"] else (f["a"] - f["h"]); clv = None
    if res == 0: continue
    for key in (f"{k}_{z['tier']}", f"{k}_{z['tier']}_{'early' if z.get('fresh') else 'late'}"):
        s2 = T[key]; s2["n"] += 1; s2["w"] += res > 0; s2["pnl"] += (d - 1) if res > 0 else -1; s2["clv"] += clv or 0
for k, s in T.items(): out["B"][k] = {"n": s["n"], "win": round(s["w"] / s["n"], 3), "roi": round(s["pnl"] / s["n"], 3), "clv_pts": round(s["clv"] / s["n"], 2)}
json.dump(out, open(f"{H}/grade.json", "w"), indent=1); print(json.dumps(out))
