import pandas as pd, glob
out = []
for f in sorted(glob.glob("cfb/espn_*.parquet")):
    d = pd.read_parquet(f, columns=["game_id", "pos_team_id", "pass", "sack", "passer_player_id", "scrimmage_play", "game_play_number", "period"])
    d = d[(d.scrimmage_play == True) & d.passer_player_id.notna() & ((d["pass"] == True) | (d.sack == True))].sort_values("game_play_number")
    g = d.groupby(["game_id", "pos_team_id"])
    x = g.passer_player_id.last().rename("last_qb").to_frame()
    x["first_qb"] = g.passer_player_id.first()
    # did the first passer still throw in the second half?
    h2 = d[d.period >= 3].groupby(["game_id", "pos_team_id"]).passer_player_id.agg(lambda s: set(s))
    x["starter_h2"] = [ (k in h2.index) and (fq in h2[k]) for k, fq in zip(x.index, x.first_qb)]
    out.append(x.reset_index())
pd.concat(out).to_parquet("cfb_qb_firstlast.parquet"); print("ok")
