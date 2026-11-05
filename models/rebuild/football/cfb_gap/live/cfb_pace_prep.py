import sys, pandas as pd, numpy as np
# usage: cfb_pace_prep.py <dir with espn_YYYY.parquet> [years, default 2014-2026]; replaces those seasons in cfb_pace_games.parquet
PB = sys.argv[1] if len(sys.argv) > 1 else "cfb"; YRS = [int(y) for y in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(2014, 2027)
out = []
for y in YRS:
    d = pd.read_parquet(f"{PB}/espn_{y}.parquet", columns=["game_id", "pos_team_id", "def_pos_team_id", "down", "period", "start.TimeSecsRem", "end.TimeSecsRem", "pos_score_diff_start"])
    d = d[d.down.notna() & d.pos_team_id.notna()]
    d["el"] = (d["start.TimeSecsRem"] - d["end.TimeSecsRem"]).clip(0, 60)
    neu = d[(d.pos_score_diff_start.abs() <= 14) & d.period.isin([1, 2, 3])]
    gp = d.groupby("game_id").size().rename("gplays")
    op = d.groupby(["game_id", "pos_team_id"]).size().rename("oplays")
    spp = neu.groupby(["game_id", "pos_team_id"]).el.agg(["sum", "size"]).rename(columns={"sum": "nsec", "size": "nn"})
    t = pd.concat([op, spp], axis=1).reset_index().rename(columns={"pos_team_id": "team"}); t = t.merge(gp, on="game_id"); t["season"] = y
    out.append(t)
import os
P_ = pd.concat(out)
if os.path.exists("cfb_pace_games.parquet"):
    old = pd.read_parquet("cfb_pace_games.parquet"); P_ = pd.concat([old[~old.season.isin(list(YRS))], P_], ignore_index=True)
P_.to_parquet("cfb_pace_games.parquet"); print(len(P_), P_.describe().T)
