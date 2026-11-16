"""Grade CBB Model A + Model B from history/runs.jsonl against finals (ESPN scoreboard, site.api.espn.com).

One rule (audit 2026-11-13, findings 1 and 7): only rows logged BEFORE the game started count. After a game is played the
ratings include it, so later rows are not predictions.
Model A: the last pre-game row. Team-score, margin and total miss vs actual, with the market's number from the same row beside
         it on the same games (J yardstick).
Model B: per tier, the FIRST pre-game row where a pick reached its tier (what J could have bet) -> win rate, ROI at the shown
         price, and CLV vs the last pre-game market number, averaged only over picks that have a later market number (clv_n).
python3 models/cbb_ab/grade.py  -> models/cbb_ab/history/grade.json"""
import json, os, sys, urllib.request, datetime as dt, collections
HERE = os.path.dirname(os.path.abspath(__file__)); H = f"{HERE}/history"
sys.path.insert(0, f"{HERE}/../model-b"); import unproven   # tier C passes graded by their unproven rec (J 2026-11-06)
runs = [json.loads(l) for l in open(f"{H}/runs.jsonl")] if os.path.exists(f"{H}/runs.jsonl") else []
fin_p = f"{H}/finals.json"; fin = json.load(open(fin_p)) if os.path.exists(fin_p) else {}
today = dt.date.today().isoformat()
for d in sorted({r["d"] for r in runs if r["d"] < today}):
    day = [v for v in fin.values() if v.get("d") == d]
    if day and all("hn" in v and "k" in v for v in day): continue   # days saved before names / start times were kept are fetched once more
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard?dates={d.replace('-', '')}&groups=50&limit=400"
    try: J = json.load(urllib.request.urlopen(url, timeout=30))
    except Exception as e: print("finals fetch failed", d, e); continue
    for ev in J.get("events", []):
        c = ev["competitions"][0]
        if not c["status"]["type"].get("completed"): continue
        sc = {x["homeAway"]: float(x["score"]) for x in c["competitors"]}
        nm = {x["homeAway"]: (x.get("team") or {}).get("location") for x in c["competitors"]}
        fin[ev["id"]] = {"d": d, "k": ev.get("date"), "h": sc.get("home"), "a": sc.get("away"), "hn": nm.get("home"), "an": nm.get("away")}
    for v in day: v.setdefault("k", None)   # a saved game ESPN no longer lists is not fetched again every run
json.dump(fin, open(fin_p, "w"))
def sc(f, r):
    """Final (home, away) score from the side order of run r. ESPN can swap home and away at a neutral site after the run
    was logged; the names saved with the final tell (finals saved before 2026-11-13 have no names and are used as they are)."""
    return (f["a"], f["h"]) if f.get("hn") and f["hn"] == r.get("a") and f.get("an") == r.get("h") else (f["h"], f["a"])
def when(x):
    """Run stamp ('2026-11-16 14:25 UTC') or ISO start time -> aware datetime, None if unreadable."""
    try:
        x = str(x).strip()
        if x.endswith(" UTC"): return dt.datetime.strptime(x, "%Y-%m-%d %H:%M UTC").replace(tzinfo=dt.timezone.utc)
        return dt.datetime.fromisoformat(x.replace("Z", "+00:00"))
    except Exception: return None
def start(gid, r):
    """Start time of a game: the final's (ESPN), else the one saved on the row. None = unknown."""
    return when((fin.get(gid) or {}).get("k")) or when(r.get("k"))
def pregame(r):
    """True when the row was logged before the game started. With no start time known, a row counts only when it was logged
    before the game's date began in US Eastern time (05:00 UTC), which is before any game that day."""
    t = when(r["t"]); k = start(r["id"], r)
    if t is None: return False
    if k is not None: return t < k
    return t < dt.datetime.fromisoformat(r["d"]).replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=5)
first, firstU, last, dropped = {}, {}, {}, set()
for r in runs:
    if not pregame(r): dropped.add(r["id"]); continue
    last[r["id"]] = r
    for k, z in r["B"].items():
        if z["tier"] in "AB" and z.get("bet") and (r["id"], k) not in first: first[(r["id"], k)] = (r, z)
        elif z["tier"] == "C" and z.get("bet") and (r["id"], k) not in firstU and (u := unproven.grade_tier("CBB", k, z)): firstU[(r["id"], k)] = (r, dict(z, tier=u))
out = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "rule": "rows logged before the game started only",
       "games_final": sum(1 for i in fin if i in last or i in dropped), "games_graded": sum(1 for i in fin if i in last),
       "games_without_pregame_row": sum(1 for i in fin if i in dropped and i not in last), "A": {}, "B": {}}
mae = lambda v: round(sum(v) / len(v), 2) if v else None
A = collections.defaultdict(list); BK = collections.defaultdict(lambda: collections.defaultdict(list))
for i, f in fin.items():
    r = last.get(i)
    if not r: continue
    fh, fa = sc(f, r); oh, oa = r["A"]
    A["team"] += [abs(oh - fh), abs(oa - fa)]
    ms = (r["B"].get("spread") or {}).get("market"); mt = (r["B"].get("total") or {}).get("market")
    if ms is None or mt is None: continue
    lh, la = (mt + ms) / 2, (mt - ms) / 2     # the market's implied team scores (spread market is the home margin)
    A["team_l_ours"] += [abs(oh - fh), abs(oa - fa)]; A["team_l_line"] += [abs(lh - fh), abs(la - fa)]
    A["margin_ours"].append(abs((oh - oa) - (fh - fa))); A["margin_line"].append(abs(ms - (fh - fa)))
    A["total_ours"].append(abs((oh + oa) - (fh + fa))); A["total_line"].append(abs(mt - (fh + fa))); A["total_bias"].append((oh + oa) - (fh + fa))
    # every lined game by size of the line, big mismatches included (research 2026-11-13, idea 4): the back-tests had few of them
    b = "under 7" if abs(ms) < 7 else "7 to 14" if abs(ms) < 14 else "14 to 21" if abs(ms) < 21 else "21 and up"; sg = 1 if ms >= 0 else -1
    BK[b]["ours"].append(abs((oh - oa) - (fh - fa))); BK[b]["line"].append(abs(ms - (fh - fa)))
    BK[b]["short"].append(sg * ((oh - oa) - ms)); BK[b]["result"].append(sg * ((fh - fa) - ms))
if A["team"]:
    out["A"] = {"team_scores": len(A["team"]), "MAE": mae(A["team"]),
                "lined": {"games": len(A["margin_ours"]), "team_MAE": mae(A["team_l_ours"]), "line_team_MAE": mae(A["team_l_line"]),
                          "margin_MAE": mae(A["margin_ours"]), "line_margin_MAE": mae(A["margin_line"]),
                          "total_MAE": mae(A["total_ours"]), "line_total_MAE": mae(A["total_line"]), "total_bias": mae(A["total_bias"])},
                # ours_vs_line / result_vs_line: points toward the favourite (negative = short of the line)
                "by_line": {b: {"games": len(v["ours"]), "margin_MAE": mae(v["ours"]), "line_margin_MAE": mae(v["line"]), "ours_vs_line": mae(v["short"]), "result_vs_line": mae(v["result"])}
                            for b in ("under 7", "7 to 14", "14 to 21", "21 and up") if (v := BK.get(b))}}
T = collections.defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0, "clv": 0.0, "clv_n": 0})
for (gid, k), (r, z) in [*first.items(), *firstU.items()]:
    f = fin.get(gid)
    if not f: continue
    fh, fa = sc(f, r); b = z["bet"]; pt = b["pt"]; dec = 1 + (b["price"] / 100 if b["price"] > 0 else 100 / -b["price"])
    later = last[gid] if last[gid]["t"] != r["t"] else None      # CLV needs a market number from a later pre-game row
    if k == "spread":
        m = fh - fa; res = (m + pt) if z["side"] == r["h"] else (-m + pt)
        lm = later and (later["B"].get("spread") or {}).get("market"); clv = None if lm is None else ((lm - z["market"]) if z["side"] == r["h"] else (z["market"] - lm))
    elif k == "total":
        t = fh + fa; res = (t - pt) if z["side"] == "Over" else (pt - t)
        lm = later and (later["B"].get("total") or {}).get("market"); clv = None if lm is None else ((lm - z["market"]) if z["side"] == "Over" else (z["market"] - lm))
    else:   # moneyline
        res = (fh - fa) if z["side"] == r["h"] else (fa - fh); clv = None
    if res == 0: continue
    dog = (z["market"] > 0) != (z["side"] == r["h"])   # side is the market underdog (J 2026-11-02: watch the dog lean)
    big = k == "spread" and dog and abs(z["market"]) >= 20     # J 2026-11-09: big dogs (20+) graded as their own bucket
    for key in [f"{k}_{z['tier']}"] + ([f"{k}_{z['tier']}_{'dog' if dog else 'fav'}"] if k != "total" else []) + ([f"{k}_{z['tier']}_bigdog20"] if big else []):
        s2 = T[key]; s2["n"] += 1; s2["w"] += res > 0; s2["pnl"] += (dec - 1) if res > 0 else -1
        if clv is not None: s2["clv"] += clv; s2["clv_n"] += 1
for k, s in T.items(): out["B"][k] = {"n": s["n"], "win": round(s["w"] / s["n"], 3), "roi": round(s["pnl"] / s["n"], 3),
                                      "clv_pts": round(s["clv"] / s["clv_n"], 2) if s["clv_n"] else None, "clv_n": s["clv_n"]}
json.dump(out, open(f"{H}/grade.json", "w"), indent=1); print(json.dumps(out, indent=1))
