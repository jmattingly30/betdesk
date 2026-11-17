"""Rebuild the static history the daily run starts from (finished seasons only). Run once a year after the season ends.
Usage: python3 hist.py <ev dir from prep_events.py over past seasons> <last finished season, start year>
Writes data/hist.parquet (one row per team-game) and data/sums.json (league sums behind the base rates)."""
import sys, os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import lib
E, LAST = sys.argv[1], int(sys.argv[2])
s = pd.read_parquet(f"{E}/shots.parquet"); t = pd.read_parquet(f"{E}/segs.parquet")
s = s[s.season <= LAST]; t = t[t.season <= LAST]
S = lib.sums(s, t); bases = {int(y): lib.base_of(lib.pool(S, int(y))) for y in sorted(s.season.unique())}
TG = lib.team_games(s, t, bases)
os.makedirs(f"{HERE}/data", exist_ok=True)
TG.to_parquet(f"{HERE}/data/hist.parquet"); json.dump(S, open(f"{HERE}/data/sums.json", "w"))
print(len(TG), "team-games", sorted(bases))
