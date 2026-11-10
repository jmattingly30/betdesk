"""Grade CBB Model A + Model B from history/runs.jsonl against finals (ESPN scoreboard, site.api.espn.com).
Model A: team-score miss vs actual and vs the market's implied team totals (J yardstick). Model B: per tier, the FIRST
time a pick reached its tier (what J could have bet) -> win rate, ROI at the shown price, CLV vs the last market number seen.
python3 models/cbb_ab/grade.py  -> models/cbb_ab/history/grade.json"""
import json, os, sys, urllib.request, datetime as dt, collections
HERE = os.path.dirname(os.path.abspath(__file__)); H = f"{HERE}/history"
import sys; sys.path.insert(0, f"{HERE}/../model-b"); import unproven   # tier C passes graded by their unproven rec (J 2026-11-06)
runs = [json.loads(l) for l in open(f"{H}/runs.jsonl")] if os.path.exists(f"{H}/runs.jsonl") else []
fin_p = f"{H}/finals.json"; fin = json.load(open(fin_p)) if os.path.exists(fin_p) else {}
today = dt.date.today().isoformat()
for d in sorted({r["d"] for r in runs if r["d"] < today}):
    if any(v.get("d") == d for v in fin.values()): continue
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={d.replace('-', '')}&groups=50&limit=400"
    try: J = json.load(urllib.request.urlopen(url, timeout=30))
    except Exception as e: print("finals fetch failed", d, e); continue
    for ev in J.get("events", []):
        c = ev["competitions"][0]
        if not c["status"]["type"].get("completed"): continue
        sc = {x["homeAway"]: float(x["score"]) for x in c["competitors"]}
        fin[ev["id"]] = {"d": d, "h": sc.get("home"), "a": sc.get("away")}
json.dump(fin, open(fin_p, "w"))
first, firstU, last = {}, {}, {}
for r in runs:
    last[r["id"]] = r
    for k, z in r["B"].items():
        if z["tier"] in "AB" and z.get("bet") and (r["id"], k) not in first: first[(r["id"], k)] = (r, z)
        elif z["tier"] == "C" and z.get("bet") and (r["id"], k) not in firstU and (u := unproven.grade_tier("CBB", k, z)): firstU[(r["id"], k)] = (r, dict(z, tier=u))
out = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "A": {}, "B": {}}
e = [abs(last[i]["A"][0] - f["h"]) + 0 for i, f in fin.items() if i in last] + [abs(last[i]["A"][1] - f["a"]) for i, f in fin.items() if i in last]
if e: out["A"] = {"team_scores": len(e), "MAE": round(sum(e) / len(e), 2)}
T = collections.defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0, "clv": 0.0})
for (gid, k), (r, z) in [*first.items(), *firstU.items()]:
    f = fin.get(gid)
    if not f: continue
    b = z["bet"]; pt = b["pt"]; dec = 1 + (b["price"] / 100 if b["price"] > 0 else 100 / -b["price"])
    if k == "spread":
        m = f["h"] - f["a"]; res = (m + pt) if z["side"] == r["h"] else (-m + pt)
        lm = last[gid]["B"].get("spread", {}).get("market"); clv = None if lm is None else ((lm - z["market"]) if z["side"] == r["h"] else (z["market"] - lm))
    elif k == "total":
        t = f["h"] + f["a"]; res = (t - pt) if z["side"] == "Over" else (pt - t)
        lm = last[gid]["B"].get("total", {}).get("market"); clv = None if lm is None else ((lm - z["market"]) if z["side"] == "Over" else (z["market"] - lm))
    else:   # moneyline
        res = (f["h"] - f["a"]) if z["side"] == r["h"] else (f["a"] - f["h"]); clv = None
    if res == 0: continue
    dog = (z["market"] > 0) != (z["side"] == r["h"])   # side is the market underdog (J 2026-11-02: watch the dog lean)
    big = k == "spread" and dog and abs(z["market"]) >= 20     # J 2026-11-09: big dogs (20+) graded as their own bucket
    for key in [f"{k}_{z['tier']}"] + ([f"{k}_{z['tier']}_{'dog' if dog else 'fav'}"] if k != "total" else []) + ([f"{k}_{z['tier']}_bigdog20"] if big else []):
        s2 = T[key]; s2["n"] += 1; s2["w"] += res > 0; s2["pnl"] += (dec - 1) if res > 0 else -1; s2["clv"] += clv or 0
for k, s in T.items(): out["B"][k] = {"n": s["n"], "win": round(s["w"] / s["n"], 3), "roi": round(s["pnl"] / s["n"], 3), "clv_pts": round(s["clv"] / s["n"], 2)}
json.dump(out, open(f"{H}/grade.json", "w"), indent=1); print(json.dumps(out, indent=1))
