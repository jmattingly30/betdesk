"""Grade NBA Model A + Model B from history/runs.jsonl against finals (rebuild teamgames_nba.parquet, ESPN ids = slate ids).
Model A (J yardstick): team-score miss vs actual AND vs the market's implied team totals (last market seen before tip);
also the base rebuild means (before the component fix), so the fix is tracked live.
Model B: per tier and kind, the FIRST time a pick reached tier A/B (what J could have bet) -> win rate, ROI at the shown price,
CLV in points vs the last market number seen.  python3 models/nba_ab/grade.py -> models/nba_ab/history/grade.json"""
import json, os, datetime as dt, collections, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); H = f"{HERE}/history"; ROOT = os.path.normpath(f"{HERE}/../..")
import sys; sys.path.insert(0, f"{HERE}/../model-b"); import unproven   # tier C passes graded by their unproven rec (J 2026-11-06)
runs = [json.loads(l) for l in open(f"{H}/runs.jsonl")] if os.path.exists(f"{H}/runs.jsonl") else []
tg = pd.read_parquet(f"{ROOT}/models/rebuild/basketball/data/teamgames_nba.parquet", columns=["game_id", "ha", "pts"])
P = tg.pivot_table(index="game_id", columns="ha", values="pts", aggfunc="first")
fin = {str(g): {"h": float(r["home"]), "a": float(r["away"])} for g, r in P.iterrows() if r.notna().all()}
first, firstU, last = {}, {}, {}
for r in runs:
    last[r["id"]] = r
    for k, z in r["B"].items():
        if z["tier"] in "AB" and z.get("bet") and (r["id"], k) not in first: first[(r["id"], k)] = (r, z)
        elif z["tier"] == "C" and z.get("bet") and (r["id"], k) not in firstU and (u := unproven.grade_tier("NBA", k, z)): firstU[(r["id"], k)] = (r, dict(z, tier=u))
out = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "A": {}, "B": {}}
eA, eB, eL, closer = [], [], [], 0
for i, f in fin.items():
    r = last.get(i)
    if not r: continue
    for j, side in ((0, "h"), (1, "a")): eA.append(abs(r["A"][j] - f[side])); eB.append(abs(r["base"][j] - f[side]))
    sp, tt = r["B"].get("spread"), r["B"].get("total")
    if sp and tt and sp.get("market") is not None and tt.get("market") is not None:
        ih, ia = (tt["market"] + sp["market"]) / 2, (tt["market"] - sp["market"]) / 2
        for j, side, im in ((0, "h", ih), (1, "a", ia)):
            eL.append(abs(im - f[side])); closer += abs(r["A"][j] - f[side]) < abs(im - f[side])
if eA: out["A"] = {"team_scores": len(eA), "MAE": round(sum(eA) / len(eA), 2), "base_MAE": round(sum(eB) / len(eB), 2)}
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
