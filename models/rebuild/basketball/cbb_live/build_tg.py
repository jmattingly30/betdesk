"""Rebuild team-games (prep.py cbb()) from the live CBB copy (cbb_live/cbb/data, includes the season in progress)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import prep
prep.M = HERE   # prep.cbb() reads {M}/cbb/data
t = prep.finish(prep.cbb()); t.to_parquet(f"{HERE}/teamgames_cbb.parquet")
print("teamgames", t.shape, t.season.min(), t.season.max(), (t.season == t.season.max()).sum(), "rows this season")
