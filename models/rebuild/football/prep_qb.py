"""QB-game table: every QB's dropbacks (passes, sacks, scrambles) and EPA per game. -> data/qb_games_nfl.parquet"""
import sys, os, glob, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); dl = sys.argv[1]; out = []
for f in sorted(glob.glob(f"{dl}/pbp_*.parquet")):
    d = pd.read_parquet(f, columns=["game_id", "posteam", "qb_dropback", "passer_player_id", "rusher_player_id", "epa", "season"])
    d = d[(d.qb_dropback == 1) & d.epa.notna()].copy()
    d["qb"] = d.passer_player_id.fillna(d.rusher_player_id)
    out.append(d[d.qb.notna()].groupby(["game_id", "posteam", "qb"]).agg(db=("epa", "size"), epa=("epa", "sum")).reset_index())
Q = pd.concat(out).rename(columns={"posteam": "team"}); Q.to_parquet(f"{HERE}/data/qb_games_nfl.parquet"); print(len(Q))
