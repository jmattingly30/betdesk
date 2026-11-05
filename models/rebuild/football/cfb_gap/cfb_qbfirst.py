import pandas as pd, glob
out = []
for f in sorted(glob.glob("cfb/espn_*.parquet")):
    d = pd.read_parquet(f, columns=["game_id", "pos_team_id", "pass", "sack", "passer_player_id", "scrimmage_play", "game_play_number"])
    d = d[(d.scrimmage_play == True) & d.passer_player_id.notna() & ((d["pass"] == True) | (d.sack == True))]
    out.append(d.sort_values("game_play_number").groupby(["game_id", "pos_team_id"]).passer_player_id.first().reset_index())
pd.concat(out).to_parquet("cfb_qb_first.parquet"); print("ok")
