"""Grade the NHL sim's pre-game rows. Usage: python3 grade.py <scratch dir used by live.py>
For each finished game: the last row logged before its start time. Team goals (the overtime or shootout goal counts) against
the sim, and on the games that had a line, against the team goals implied by the moneyline and total at that time.
Writes history/grade.json."""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
SP = sys.argv[1]
sch = pd.read_parquet(f"{SP}/nhlsim_sched.parquet"); fin = sch[sch.game_state.isin(["OFF", "FINAL"])]
fin = {str(int(g)): (h, a) for g, h, a in zip(fin.game_id, fin.home_score, fin.away_score)}
last = {}
try:
    for ln in open(f"{HERE}/history/log.jsonl"):
        r = json.loads(ln)
        if r["made"] < r["k"]: last[r["id"]] = r      # file is in time order, so the last pre-game row wins
except FileNotFoundError: pass
rows = []
for gid, r in last.items():
    if gid not in fin: continue
    for side, i in (("h", 0), ("a", 1)):
        rows.append({"y": float(fin[gid][i]), "p": r[side + "s"], "ln": r.get("l" + side), "b2b": r["b2b"][i], "d": r["d"],
                     "pt": r["hs"] + r["as"], "yt": float(sum(fin[gid])), "tl": r.get("lt"), "pw": r["pw"], "hw": float(fin[gid][0] > fin[gid][1]), "home": i == 0,
                     "team": r[side], "id": gid, "h": r["h"], "a": r["a"], "ps": [r["hs"], r["as"]], "fs": list(map(int, fin[gid])), "lhl": [r.get("lh"), r.get("la")]})
D = pd.DataFrame(rows); out = {"games": len(D) // 2}
if len(D): D["ln"] = pd.to_numeric(D.ln); D["tl"] = pd.to_numeric(D.tl)
def blk(d):
    o = {"n": len(d), "sim_miss": round(float((d.p - d.y).abs().mean()), 3), "sim_bias": round(float((d.p - d.y).mean()), 3)}
    L = d.dropna(subset=["ln"])
    if len(L): o.update(lined=len(L), sim_miss_lined=round(float((L.p - L.y).abs().mean()), 3), line_miss=round(float((L.ln - L.y).abs().mean()), 3),
                        line_bias=round(float((L.ln - L.y).mean()), 3))
    return o
if len(D):
    out["team_goals"] = blk(D); out["second_night"] = blk(D[D.b2b == 1]) if (D.b2b == 1).any() else {"n": 0}
    H = D[D.home]; out["total"] = {"n": len(H), "sim_miss": round(float((H.pt - H.yt).abs().mean()), 3)}
    T = H.dropna(subset=["tl"])
    if len(T): out["total"].update(lined=len(T), sim_miss_lined=round(float((T.pt - T.yt).abs().mean()), 3), line_miss=round(float((T.tl - T.yt).abs().mean()), 3))
    out["winners"] = round(float(((H.pw > .5) == (H.hw == 1)).mean()), 3); out["first"] = D.d.min(); out["last"] = D.d.max()
    # per team (J 2026-10-18 model tracking: scored vs our projection, team by team) and the latest graded games for the desk
    out["teams"] = {tm: {"n": len(d), "proj": round(float(d.p.mean()), 2), "act": round(float(d.y.mean()), 2), "miss": round(float((d.p - d.y).abs().mean()), 2),
                         **({"line_miss": round(float((d.dropna(subset=["ln"]).ln - d.dropna(subset=["ln"]).y).abs().mean()), 2)} if d.ln.notna().any() else {})}
                    for tm, d in D.groupby("team")}
    out["recent"] = [{"id": r.id, "d": r.d, "h": r.h, "a": r.a, "ps": r.ps, "fs": r.fs, "l": r.lhl, "tl": r.tl if pd.notna(r.tl) else None}
                     for r in H.sort_values("d").tail(30).itertuples()][::-1]
out["made"] = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
json.dump(out, open(f"{HERE}/history/grade.json", "w"), indent=1)
try:      # the desk reads the grade from nhl_sim.json (live.py also copies it in on every run)
    P = f"{os.path.abspath(f'{HERE}/../..')}/app/data/nhl_sim.json"; j = json.load(open(P)); j["grade"] = out; json.dump(j, open(P, "w"))
except Exception as e: print("::warning::grade not copied into nhl_sim.json:", e)
print("NHL sim grade:", json.dumps(out))
