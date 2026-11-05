import pandas as pd, numpy as np
out = []
for y in range(2014, 2027):
    d = pd.read_parquet(f"cfb/espn_{y}.parquet", columns=["game_id", "pos_team_id", "def_pos_team_id", "down", "period", "start.TimeSecsRem", "end.TimeSecsRem", "pos_score_diff_start"])
    d = d[d.down.notna() & d.pos_team_id.notna()]
    d["el"] = (d["start.TimeSecsRem"] - d["end.TimeSecsRem"]).clip(0, 60)
    neu = d[(d.pos_score_diff_start.abs() <= 14) & d.period.isin([1, 2, 3])]
    gp = d.groupby("game_id").size().rename("gplays")
    op = d.groupby(["game_id", "pos_team_id"]).size().rename("oplays")
    spp = neu.groupby(["game_id", "pos_team_id"]).el.agg(["sum", "size"]).rename(columns={"sum": "nsec", "size": "nn"})
    t = pd.concat([op, spp], axis=1).reset_index().rename(columns={"pos_team_id": "team"}); t = t.merge(gp, on="game_id"); t["season"] = y
    out.append(t)
P_ = pd.concat(out); P_.to_parquet("cfb_pace_games.parquet"); print(len(P_), P_.describe().T)
