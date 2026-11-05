"""Hockey benchmarks (no NHL team model exists today):
  goals_kalman  the classic approach: each team's attack and defense on log goals, walk-forward (Kalman)
  league        season-to-date league average goals + home edge
Writes data/baseline_nhl_{name}.parquet."""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
tg = pd.read_parquet(f"{HERE}/data/teamgames_nhl.parquet")
h = tg[tg.ha == "home"].set_index("game_id"); a = tg[tg.ha == "away"].set_index("game_id")
g = h.join(a, rsuffix="_a", how="inner").reset_index().sort_values(["date", "game_id"]).reset_index(drop=True)
teams = pd.unique(pd.concat([g.team, g.team_a])); ix = {t: i for i, t in enumerate(teams)}
A = np.zeros(len(teams)); D = np.zeros(len(teams)); PA = np.full(len(teams), 0.02); PD = PA.copy()
mu, hca = np.log(2.8), 0.05; rows = []; cur = None; lg_sum = lg_n = 0
for d, gd in g.groupby("date", sort=True):
    if gd.season.iloc[0] != cur:
        if cur is not None: A *= 0.6; D *= 0.6; PA = PA * 0.36 + 0.01; PD = PD * 0.36 + 0.01; lg_sum = lg_n = 0
        cur = gd.season.iloc[0]
    for r in gd.itertuples():
        i, j = ix[r.team], ix[r.team_a]
        lgm = (lg_sum / lg_n) if lg_n > 50 else np.exp(mu)
        rows.append({"game_id": r.game_id, "season": r.season, "date": d, "home": r.team, "away": r.team_a,
                     "pred_h": np.exp(mu + hca + A[i] + D[j]), "pred_a": np.exp(mu - hca + A[j] + D[i]),
                     "lg_h": lgm * np.exp(0.05), "lg_a": lgm * np.exp(-0.05), "act_h": r.gf, "act_a": r.gf_a})
    for r in gd.itertuples():
        i, j = ix[r.team], ix[r.team_a]
        for (o, df, side, y) in ((i, j, 1, r.gf), (j, i, -1, r.gf_a)):
            pr = mu + side * hca + A[o] + D[df]; lam = np.exp(pr)
            v = (y - lam) / lam; Rr = 1 / lam   # Poisson score step on the log scale
            S = PA[o] + PD[df] + Rr; ka, kd = PA[o] / S, PD[df] / S
            A[o] += ka * v; D[df] += kd * v; PA[o] *= 1 - ka; PD[df] *= 1 - kd
            mu += 0.002 * v; hca += 0.001 * side * v
            lg_sum += y; lg_n += 1
        PA += 0.00002; PD += 0.00002
p = pd.DataFrame(rows)
p[["game_id", "season", "date", "home", "away", "pred_h", "pred_a", "act_h", "act_a"]].to_parquet(f"{HERE}/data/baseline_nhl_goals.parquet")
p.assign(pred_h=p.lg_h, pred_a=p.lg_a)[["game_id", "season", "date", "home", "away", "pred_h", "pred_a", "act_h", "act_a"]].to_parquet(f"{HERE}/data/baseline_nhl_league.parquet")
sys.path.insert(0, os.path.dirname(HERE)); import harness as Hh
t = p[p.season >= 2022]
print(pd.DataFrame({"goals_kalman": Hh.score(t), "league": Hh.score(t.assign(pred_h=t.lg_h, pred_a=t.lg_a))}).loc[["n", "team_mae", "team_rmse", "team_bias", "margin_mae", "total_mae"]])
