"""NBA full stack (lineup + schedule + luck + recentering) on the rebuild-owned live lineup feed, through the latest
finished game. Writes data/live/stack_nba_live.parquet (walk-forward predictions; the lineup is the one known
before tip-off)."""
import sys, os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("LINEUP_AGG", f"{HERE}/data/live/lineup_agg.parquet")
import engine as E, ctx
mods, pf = ctx.STACK["nba"]
prm = {**E.DEFAULT, **json.load(open(f"{HERE}/{pf}"))}
tg = pd.read_parquet(f"{HERE}/data/teamgames_nba.parquet")
p, B, names = ctx.model(tg, prm, mods, 2017)
p.to_parquet(f"{HERE}/data/live/stack_nba_live.parquet"); print("rows", len(p), "2027 games", int((p.season == 2027).sum()))
