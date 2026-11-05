import sys, os, json, time, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E
prm = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
tag = sys.argv[2] if len(sys.argv) > 2 else "frame0"
tg = pd.read_parquet(f"{HERE}/data/teamgames_nhl.parquet")
t0 = time.time(); p, R = E.run(tg, prm); print("secs", round(time.time() - t0, 1))
p.to_parquet(f"{HERE}/data/{tag}_nhl.parquet")
b = pd.read_parquet(f"{HERE}/data/baseline_nhl_goals.parquet")
P, B = p[p.season >= 2022], b[b.season >= 2022]
print(pd.DataFrame({"goals_kalman": H.score(B), tag: H.score(P)}).loc[["n", "team_mae", "team_rmse", "team_bias", "margin_mae", "margin_rmse", "total_mae", "total_rmse"]].to_string())
for l in ("team_abs", "margin_abs", "total_abs", "margin_sq"): print(H.compare(B, P, l))
