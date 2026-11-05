"""NFL benchmarks: classic points Kalman (team offense/defense on points, home field, season carry), plus the closing
line's implied team scores as a yardstick. Writes data/baseline_nfl_points.parquet."""
import sys, os, itertools, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
import harness as H
SPORT = os.environ.get("SPORT", "nfl"); TUNE = (2008, 2017) if SPORT == "nfl" else (2015, 2019); TEST = 2018 if SPORT == "nfl" else 2021
T = pd.read_parquet(f"{HERE}/data/teamgames_{SPORT}.parquet")
h = T[T.is_home == 1].set_index("game_id"); a = T[T.is_home == 0].set_index("game_id")
g = h.join(a, rsuffix="_a", how="inner").reset_index().sort_values(["date", "game_id"]).reset_index(drop=True)
g["wk"] = g.season * 100 + g.week
def run(k=0.08, carry=0.6, hfa=1.2):
    O, D = {}, {}; mu = 22.0 if SPORT == "nfl" else 28.0; rows = []; cur = None
    for wk, gw in g.groupby("wk", sort=True):
        if gw.season.iloc[0] != cur:
            for t in O: O[t] *= carry; D[t] *= carry
            cur = gw.season.iloc[0]
        for r in gw.itertuples():
            s = 0 if r.location == "Neutral" else 1
            ph = mu + O.get(r.team, 0) + D.get(r.team_a, 0) + s * hfa; pa = mu + O.get(r.team_a, 0) + D.get(r.team, 0) - s * hfa
            rows.append((r.game_id, r.season, r.week, r.date, r.team, r.team_a, ph, pa, r.final, r.final_a, r.spread_line, r.total_line))
        for r in gw.itertuples():
            s = 0 if r.location == "Neutral" else 1
            ph = mu + O.get(r.team, 0) + D.get(r.team_a, 0) + s * hfa; pa = mu + O.get(r.team_a, 0) + D.get(r.team, 0) - s * hfa
            for off, de, e in ((r.team, r.team_a, r.final - ph), (r.team_a, r.team, r.final_a - pa)):
                O[off] = O.get(off, 0) + k * e; D[de] = D.get(de, 0) + k * e
                mu += 0.002 * e
    return pd.DataFrame(rows, columns=["game_id", "season", "week", "date", "home", "away", "pred_h", "pred_a", "act_h", "act_a", "line_margin", "line_total"])
best = None
for k, c, hf in itertools.product([0.04, 0.06, 0.08, 0.1], [0.4, 0.6, 0.8], [1.0, 1.5, 2.0]):
    p = run(k, c, hf); s = H.score(p[(p.season >= TUNE[0]) & (p.season <= TUNE[1])])
    obj = s["team_mae"] + 0.5 * s["margin_rmse"]
    if best is None or obj < best[0]: best = (obj, (k, c, hf))
print("best", best)
p = run(*best[1]); p.to_parquet(f"{HERE}/data/baseline_{SPORT}_points.parquet")
t = p[p.season >= TEST].dropna(subset=["line_margin", "line_total"])
cl = t.assign(pred_h=(t.line_total + t.line_margin) / 2, pred_a=(t.line_total - t.line_margin) / 2)
print(pd.DataFrame({"points_kalman": H.score(t), "close": H.score(cl)}).to_string())
