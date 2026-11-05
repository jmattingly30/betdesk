import sys, os, json, time, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E
prm = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].startswith("{") else (json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {})
tag = sys.argv[2] if len(sys.argv) > 2 else "frame0"
SPORT = os.environ.get("SPORT", "nfl"); TUNE = (2008, 2017) if SPORT == "nfl" else (2015, 2019); TEST = 2018 if SPORT == "nfl" else 2021
if SPORT == "cfb": E.MAP.update(json.load(open(f"{HERE}/data/map_cfb.json")))
T = pd.read_parquet(f"{HERE}/data/teamgames_{SPORT}.parquet")
t0 = time.time(); p, R = E.run(T, prm); print("secs", round(time.time() - t0, 1))
p.to_parquet(f"{HERE}/data/{tag}_{SPORT}.parquet")
b = pd.read_parquet(f"{HERE}/data/baseline_{SPORT}_points.parquet")
for lo, hi, nm in ((TUNE[0], TUNE[1], "tune"), (TEST, 2099, "test")):
    P, B = p[(p.season >= lo) & (p.season <= hi)], b[(b.season >= lo) & (b.season <= hi)]
    print(nm, pd.DataFrame({"points_kalman": H.score(B), tag: H.score(P)}).loc[["n", "team_mae", "team_bias", "margin_mae", "margin_rmse", "total_mae", "total_rmse", "total_bias"]].T.to_string())
