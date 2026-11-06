"""Grade NFL Model A + Model B from history/runs.jsonl against finals (nflverse games.csv).
Model A (J yardstick): team-score miss vs actual AND vs the market's implied team totals (last market seen before kickoff).
Model B: per tier and kind, the FIRST time a pick reached tier A/B (what J could have bet) -> win rate, ROI at the shown price,
CLV in points vs the last market number seen. python3 models/nfl_ab/grade.py [games.csv] -> models/nfl_ab/history/grade.json"""
import json, os, sys, urllib.request, datetime as dt, collections, csv, io
HERE = os.path.dirname(os.path.abspath(__file__)); H = f"{HERE}/history"
import sys; sys.path.insert(0, f"{HERE}/../model-b"); import unproven   # tier C passes graded by their unproven rec (J 2026-11-06)
runs = [json.loads(l) for l in open(f"{H}/runs.jsonl")] if os.path.exists(f"{H}/runs.jsonl") else []
src = sys.argv[1] if len(sys.argv) > 1 else "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
txt = open(src).read() if os.path.exists(src) else urllib.request.urlopen(src, timeout=60).read().decode()
fin = {r["game_id"]: {"h": float(r["home_score"]), "a": float(r["away_score"])} for r in csv.DictReader(io.StringIO(txt)) if r["home_score"] not in ("", "NA")}
first, firstU, last = {}, {}, {}
for r in runs:
    last[r["id"]] = r
    for k, z in r["B"].items():
        if z["tier"] in "AB" and z.get("bet") and (r["id"], k) not in first: first[(r["id"], k)] = (r, z)
        elif z["tier"] == "C" and z.get("bet") and (r["id"], k) not in firstU and (u := unproven.grade_tier("NFL", k, z)): firstU[(r["id"], k)] = (r, dict(z, tier=u))
out = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "A": {}, "B": {}}
eA, eL, closer = [], [], 0
for i, f in fin.items():
    r = last.get(i)
    if not r: continue
    for j, side in ((0, "h"), (1, "a")): eA.append(abs(r["A"][j] - f[side]))
    sp, tt = r["B"].get("spread"), r["B"].get("total")
    if sp and tt and sp.get("market") is not None and tt.get("market") is not None:
        ih, ia = (tt["market"] + sp["market"]) / 2, (tt["market"] - sp["market"]) / 2
        for j, side, im in ((0, "h", ih), (1, "a", ia)):
            eL.append(abs(im - f[side])); closer += abs(r["A"][j] - f[side]) < abs(im - f[side])
if eA: out["A"] = {"team_scores": len(eA), "MAE": round(sum(eA) / len(eA), 2)}
if eL: out["A"].update({"line_team_scores": len(eL), "line_MAE": round(sum(eL) / len(eL), 2), "A_closer_share": round(closer / len(eL), 3)})
T = collections.defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0, "clv": 0.0})
for (gid, k), (r, z) in [*first.items(), *firstU.items()]:
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
    s2 = T[f"{k}_{z['tier']}"]; s2["n"] += 1; s2["w"] += res > 0; s2["pnl"] += (d - 1) if res > 0 else -1; s2["clv"] += clv or 0
for k, s in T.items(): out["B"][k] = {"n": s["n"], "win": round(s["w"] / s["n"], 3), "roi": round(s["pnl"] / s["n"], 3), "clv_pts": round(s["clv"] / s["n"], 2)}
json.dump(out, open(f"{H}/grade.json", "w"), indent=1); print(json.dumps(out))
