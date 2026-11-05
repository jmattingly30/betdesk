import pandas as pd, numpy as np, glob
out = []
for f in sorted(glob.glob("cfb/espn_*.parquet")):
    d = pd.read_parquet(f, columns=["game_id", "season", "pos_team_id", "pass", "sack", "EPA", "passer_player_id", "scrimmage_play", "wp_before"])
    d = d[(d.scrimmage_play == True) & d.passer_player_id.notna() & d.EPA.notna() & ((d["pass"] == True) | (d.sack == True))]
    g = d.groupby(["game_id", "season", "pos_team_id", "passer_player_id"]).agg(db=("EPA", "size"), epa=("EPA", "sum")).reset_index()
    out.append(g); print(f, len(g), flush=True)
Q = pd.concat(out); Q.to_parquet("cfb_qb_games.parquet"); print(len(Q))
