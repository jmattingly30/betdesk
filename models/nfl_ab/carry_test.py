"""NFL Model A test 1 (J 2026-11-05 rollout): season-start offense carry by play-caller / QB case.
Research (research/model-a-sim-2026-10/nfl/layers.json B_caller_qb): year-to-year offense EPA/play r = same caller + same QB .46,
new caller + same QB .35, same caller + new QB .20, both new -.01 -> "offense efficiency carries only if QB or play-caller stays".
Engine today carries every offense by the same 0.7. Here each offense's carried rating is rescaled by case at season start.
Week-1 starter = known at season start (depth chart); last season's QB = most starts. Graded on frame level, 2018+.
python3 carry_test.py -> carry_test.json"""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); FB = os.path.normpath(f"{HERE}/../rebuild/football")
sys.path.insert(0, FB); sys.path.insert(0, os.path.dirname(FB))
import harness as H, engine as E
C = pd.read_csv(f"{HERE}/../nfl-player/data/play_callers.csv"); caller = {(r.season, r.team): r.caller for r in C.itertuples()}
T = pd.read_parquet(f"{FB}/data/teamgames_nfl.parquet"); prm = json.load(open(f"{FB}/data/tuned_nfl.json"))
T["qb"] = np.where(T.is_home == 1, T.home_qb_id, T.away_qb_id)
main_qb = T.groupby(["season", "team"]).qb.agg(lambda s: s.value_counts().index[0]).to_dict()
wk1_qb = T.sort_values("date").groupby(["season", "team"]).qb.first().to_dict()
R_CASE = {(1, 1): .462, (0, 1): .35, (1, 0): .204, (0, 0): -.008}
N_CASE = {(1, 1): 123, (0, 1): 78, (1, 0): 64, (0, 0): 55}
R_ALL = sum(R_CASE[k] * N_CASE[k] for k in R_CASE) / sum(N_CASE.values())
OFF = [0, 1, 2, 3]
def case(season, t):
    c_new, c_old = caller.get((season, t)), caller.get((season - 1, t))
    q_new, q_old = wk1_qb.get((season, t)), main_qb.get((season - 1, t))
    if None in (c_new, c_old, q_new, q_old): return None
    return (int(c_new == c_old), int(q_new == q_old))
def hook(strength, floor):
    def season_start(R, season):
        for t, i in R.idx.items():
            k = case(season, t)
            if k is None: continue
            if strength < 0:            # cut-only: only offenses that lost BOTH QB and caller carry less (to half)
                if k != (0, 0): continue
                f = 0.5
            else: f = 1 + strength * (max(R_CASE[k], floor) / R_ALL - 1)      # strength 1 = full research ratio
            R.O[i, OFF] *= f
            if f < 1: R.PO[i, OFF] += (1 - f) * 0.2 * R.var0[OFF]   # less sure about an offense that lost its QB/caller
    return {"season_start": season_start}
base, _ = E.run(T, prm)
out = {"r_all": round(R_ALL, 3), "cases_2018plus": {}, "runs": {}}
cs = [case(s, t) for s, t in T[T.season >= 2018][["season", "team"]].drop_duplicates().itertuples(index=False)]
out["cases_2018plus"] = {str(k): cs.count(k) for k in R_CASE}
early = lambda d: d[(d.season >= 2018) & (d.week <= 6)]
for strength, floor in ((0.5, 0.0), (1.0, 0.0), (1.0, 0.1), (0.75, 0.05), (-1, 0.0)):
    p, _ = E.run(T, prm, hook(strength, floor)); tag = f"s{strength}_f{floor}"
    out["runs"][tag] = {nm: {loss: H.compare(sel(base), sel(p), loss) for loss in ("team_abs", "total_abs", "margin_sq")}
                        for nm, sel in (("2018+ all weeks", lambda d: d[d.season >= 2018]), ("2018+ weeks 1-6", early))}
    print(tag, {nm: (v["team_abs"]["delta"], v["team_abs"]["t"]) for nm, v in out["runs"][tag].items()}, flush=True)
    if tag == "s1.0_f0.0": p.to_parquet(f"/tmp/nfl_carry_best.parquet")
base.to_parquet("/tmp/nfl_carry_base.parquet")
json.dump(out, open(f"{HERE}/carry_test.json", "w"), indent=1)
