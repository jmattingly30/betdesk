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
                     "pt": r["hs"] + r["as"], "yt": float(sum(fin[gid])), "tl": r.get("lt"), "pw": r["pw"], "hw": float(fin[gid][0] > fin[gid][1]), "home": i == 0})
D = pd.DataFrame(rows); out = {"games": len(D) // 2}
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
json.dump(out, open(f"{HERE}/history/grade.json", "w"), indent=1); print("NHL sim grade:", json.dumps(out))
