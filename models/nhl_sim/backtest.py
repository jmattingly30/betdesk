"""Walk-forward test of the sim with and without the rest step, on past seasons.
Usage: python3 backtest.py <ev dir from prep_events.py> <dir with nhl_schedule files as sched_YYYY.parquet> [N]
The rest multiplier for each season is fitted on the other seasons only. Writes backtest.json here."""
import sys, os, glob, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib, sim
E, SD = sys.argv[1], sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
s = pd.read_parquet(f"{E}/shots.parquet"); t = pd.read_parquet(f"{E}/segs.parquet"); g = pd.read_parquet(f"{E}/games.parquet")
S = lib.sums(s, t); seasons = sorted(int(x) for x in s.season.unique())
bases = {y: lib.base_of(lib.pool(S, y)) for y in seasons}
TG = lib.team_games(s, t, bases); R, *_ = lib.rate(TG)
sch = pd.concat([pd.read_parquet(f, columns=["game_id", "game_date"]) for f in glob.glob(f"{SD}/sched_*.parquet")]).drop_duplicates("game_id")
g = g.drop(columns=["game_date"]).merge(sch, on="game_id", how="left"); g["date"] = pd.to_datetime(g.game_date)
# back-to-back from every game a team played (regular season and playoffs)
tl = pd.concat([g[["game_id", "season", "date", "home"]].rename(columns={"home": "team"}), g[["game_id", "season", "date", "away"]].rename(columns={"away": "team"})]).sort_values("date")
tl["b2b"] = (tl.groupby(["team", "season"]).date.diff().dt.days == 1).astype(float)
b = tl.set_index(["game_id", "team"]).b2b
g = g[(g.st == "R") & (g.season >= 2022) & g.date.notna()].reset_index(drop=True)
g["b2b_h"] = b.reindex(list(zip(g.game_id, g.home))).values; g["b2b_a"] = b.reindex(list(zip(g.game_id, g.away))).values
g["act_h"] = g.hg + (g.so & (g.h_so > g.a_so)); g["act_a"] = g.ag + (g.so & (g.a_so > g.h_so))
rr = lambda side: R[R.side == side].set_index("game_id").reindex(g.game_id).reset_index(drop=True)
def run(G):
    return pd.concat([sim.run(G.iloc[i:i + 1500].reset_index(drop=True), lib.sim_base(bases), lib.PARAMS, N=N, seed=i) for i in range(0, len(G), 1500)]).reset_index(drop=True)
G0 = lib.frame(g, rr("home"), rr("away")); G0["rest_h"] = 1.0; G0["rest_a"] = 1.0
o0 = run(G0)
L = pd.DataFrame({"season": np.r_[g.season, g.season], "y": np.r_[g.act_h, g.act_a], "p0": np.r_[o0.pred_h, o0.pred_a], "b2b": np.r_[g.b2b_h, g.b2b_a]})
mult = {}
for y in sorted(g.season.unique()):
    tr = L[(L.season != y) & (L.b2b == 1)]; mult[int(y)] = float(tr.y.mean() / tr.p0.mean())
G1 = G0.copy(); m = g.season.map(mult).values
G1["rest_h"] = np.where(g.b2b_h == 1, m, 1.0); G1["rest_a"] = np.where(g.b2b_a == 1, m, 1.0)
o1 = run(G1); L["p1"] = np.r_[o1.pred_h, o1.pred_a]
def cmp(d):
    e0, e1 = (d.p0 - d.y).abs(), (d.p1 - d.y).abs(); df = e1 - e0
    return {"n": len(d), "mae_off": round(e0.mean(), 4), "mae_on": round(e1.mean(), 4), "t": round(df.mean() / (df.std() / np.sqrt(len(d))), 1) if df.std() > 0 else 0.0,
            "bias_off": round((d.p0 - d.y).mean(), 3), "bias_on": round((d.p1 - d.y).mean(), 3)}
out = {"N": N, "mult_by_held_out_season": {k: round(v, 3) for k, v in mult.items()}, "mult_all": round(float(L[L.b2b == 1].y.mean() / L[L.b2b == 1].p0.mean()), 3),
       "all": cmp(L), "b2b_team": cmp(L[L.b2b == 1]), "rested_team": cmp(L[L.b2b == 0]),
       "b2b_by_season": {int(y): cmp(d) for y, d in L[L.b2b == 1].groupby("season")}}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest.json"), "w"), indent=1); print(json.dumps(out, indent=1))
