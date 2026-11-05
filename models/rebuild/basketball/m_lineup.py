"""Module test: NBA lineup / availability layer (layer 3, rates inside the engine).
Each team-game has a lineup line (pre-game player box rates x minutes, from nba-team/lineup.py). The shift for a
game = how tonight's lineup differs from the lineups the team's ratings were built on (running average of the
lineups it actually played), mapped to each possession component by ridge weights fit on earlier seasons only.
Prediction uses the lineup known before the game (continuing absences); the rating update uses who actually played.
Usage: python3 m_lineup.py tuned_params.json [halflife_games] [fit_through_season]"""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E
prm = {**E.DEFAULT, **json.load(open(sys.argv[1]))}
HL = float(sys.argv[2]) if len(sys.argv) > 2 else 15
FIT_TO = int(sys.argv[3]) if len(sys.argv) > 3 else 2017
TEST_FROM = FIT_TO + 1

tg = pd.read_parquet(f"{HERE}/data/teamgames_nba.parquet")
L = pd.read_parquet(f"{HERE}/../../nba-team/data/lineup_agg.parquet")
def feats(L, suf):
    g = lambda c: L[f"r_{c}{suf}"]
    return pd.DataFrame({"pts": g("pts"), "fta": g("fta"), "s3": g("fg3a") / g("fga"), "p3": g("fg3m") / g("fg3a"),
                         "tov": g("tov"), "oreb": g("oreb"), "ast": g("ast"), "dreb": g("dreb"), "stl": g("stl"),
                         "blk": g("blk"), "pf": g("pf"), "rapm": g("rapm")})
FA, FK, FF = feats(L, ""), feats(L, "_known"), feats(L, "_full")
# value lost to absences (healthy rotation minus tonight's): box rates undervalue stars, so give it its own weight
for c in ("pts", "rapm"):
    FA["lost_" + c] = FF[c] - FA[c]; FK["lost_" + c] = FF[c] - FK[c]
F = list(FA.columns)
L = pd.concat([L[["game_id", "team"]], FA.add_prefix("a_"), FK.add_prefix("k_")], axis=1)
L = L.merge(tg[["game_id", "team", "date"]], on=["game_id", "team"]).sort_values(["team", "date"])
# running average of lineups actually played, before this game (exponential, half-life in games)
alpha = 1 - 0.5 ** (1 / HL)
base = L.groupby("team")[[f"a_{c}" for c in F if not c.startswith("lost_")]].transform(lambda s: s.shift(1).ewm(alpha=alpha, ignore_na=True).mean())
for c in F:
    if c.startswith("lost_"):
        L[f"dk_{c}"] = L[f"k_{c}"]; L[f"da_{c}"] = L[f"a_{c}"]; continue
    b = base[f"a_{c}"].fillna(L[f"a_{c}"])
    L[f"dk_{c}"] = L[f"k_{c}"] - b; L[f"da_{c}"] = L[f"a_{c}"] - b
L = L.fillna(0.0)
sd = {c: L[f"da_{c}"].std() for c in F}
for c in F: L[f"dk_{c}"] /= sd[c]; L[f"da_{c}"] /= sd[c]
LI = L.set_index(["game_id", "team"])

g = E.games_from_teamgames(tg)
def X(kind, own, opp):
    a = LI.reindex(list(zip(g.game_id, own)))[[f"{kind}_{c}" for c in F]].values
    b = LI.reindex(list(zip(g.game_id, opp)))[[f"{kind}_{c}" for c in F]].values
    return np.nan_to_num(np.hstack([a, b]))   # own lineup, then opponent's lineup
XA_h, XA_a = X("da", g.team, g.team_a), X("da", g.team_a, g.team)
XK_h, XK_a = X("dk", g.team, g.team_a), X("dk", g.team_a, g.team)

def fit(p, B_prev, lam=50.0):
    """Ridge per component: residual logit (actual - predicted incl. current shift) on lineup deltas (actual lineups)."""
    m = (g.season <= FIT_TO).values
    B = np.zeros((2 * len(F), len(E.COMP)))
    for k, c in enumerate(E.COMP):
        ys, xs, ws = [], [], []
        for side, XA in (("", XA_h), ("_a", XA_a)):
            hp = "h_" if side == "" else "a_"
            pred = E.logit(p[hp + c].values)
            y = E.logit(g["y_" + c + side].values) - pred + XA @ B_prev[:, k]   # undo the shift already applied
            ok = m & (g["n_" + c + side].values > 0)
            ys.append(y[ok]); xs.append(XA[ok]); ws.append(np.minimum(g["n_" + c + side].values[ok], 200))
        y, x, w = np.concatenate(ys), np.vstack(xs), np.concatenate(ws)
        y = y - np.average(y, weights=w)
        A = (x * w[:, None]).T @ x + lam * w.mean() * np.eye(x.shape[1])
        B[:, k] = np.linalg.solve(A, (x * w[:, None]).T @ y)
    return B

def shifts(B):
    ph, pa, oh, oa = XK_h @ B, XK_a @ B, XA_h @ B, XA_a @ B
    return {gid: (ph[i], pa[i], oh[i], oa[i]) for i, gid in enumerate(g.game_id)}

OBS = os.environ.get("LU_OBS", "0") == "1"   # also net who-actually-played out of the rating update (tested: no help)
p0, _ = E.run(tg, prm)
B = fit(p0, np.zeros((2 * len(F), len(E.COMP))))   # one pass (re-fitting on shifted residuals made it worse)
sh = shifts(B)
if not OBS: sh = {k: (v[0], v[1], 0 * v[0], 0 * v[1]) for k, v in sh.items()}
p, _ = E.run(tg, prm, shifts=sh)
print("HL", HL, "obs", OBS, "fit-season team MAE", round(H.score(p[(p.season <= FIT_TO) & (p.season >= 2016)])["team_mae"], 4), flush=True)
np.save(f"{HERE}/data/lineup_B_hl{int(HL)}.npy", B)
print(pd.DataFrame(B, index=[f"own_{c}" for c in F] + [f"opp_{c}" for c in F], columns=E.COMP).round(4).to_string())
p.to_parquet(f"{HERE}/data/lineup_nba_hl{int(HL)}.parquet")

base = pd.read_parquet(f"{HERE}/data/baseline_nba_known.parquet")
def test(df):
    d = df[df.game_id.isin(base.game_id) & (df.season >= TEST_FROM)].copy()
    return d
pt, p0t = test(p), test(p0)
bt = base[base.game_id.isin(pt.game_id)].set_index("game_id").loc[pt.game_id].reset_index()
print(pd.DataFrame({"today": H.score(bt), "frame": H.score(p0t), "frame+lineup": H.score(pt)}).to_string())
res = {}
for loss in ("team_abs", "margin_abs", "total_abs"):
    r1 = H.compare(p0t, pt, loss); r2 = H.compare(bt, pt, loss); res[loss] = (r1, r2)
    print("vs frame", r1); print("vs today", r2)
json.dump({k: [v[0], v[1]] for k, v in res.items()}, open(f"{HERE}/data/m_lineup_result_hl{int(HL)}.json", "w"), default=float)
