"""Module test: CBB preseason roster prior (layer 1). The frame starts each season at carry x last season's rating.
This module adds a roster adjustment to each team's starting offense/defense component ratings and pace:
gamma . roster features (returning minutes, transfers in and their old team's level, value of returning players and
of transfers from models/cbb player values). gamma is fit on seasons <= FIT_TO from the frame's residuals in each
team's first N games (what the carry-only start got wrong), own features for its offense, opponent's for its defense.
Usage: python3 m_prior.py params.json [first_n_games] [scale]"""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E
NG, SCALE = 6, 1.0
FIT_TO = 2016
CBB = os.path.abspath(f"{HERE}/../../cbb")
cwd = os.getcwd(); os.chdir(CBB); sys.path.insert(0, CBB)
import priors as P
rs = pd.read_csv("data/cbb_ratings_by_season.csv")
final = {y: d.set_index("team")[["O", "D", "P", "net"]] for y, d in rs.groupby("season")}
X = P.team_features(final, pd.read_parquet("data/cbb_roster_links.parquet")); os.chdir(cwd)
RF = ["ret", "xfer_min", "xfer_q", "r_val", "r_val_lvl", "xfer_val", "new_d1"]
X = X[X.comp == "O"].assign(team=lambda d: d.team.astype(int)).set_index(["season", "team"])[RF].astype(float)
X = X.fillna(0.0); mu, sd = X.mean(), X.std(); Z = (X - mu) / sd


def prior_hook(tg, prm, p0=None, NG=6, SCALE=1.0, FIT_TO=2016, verbose=False):
    g = E.games_from_teamgames(tg)
    if p0 is None: p0, _ = E.run(tg, prm)
    # game number of each side within its season
    seq = pd.concat([g[["game_id", "season", "date", "team"]].rename(columns={"team": "t"}),
                     g[["game_id", "season", "date", "team_a"]].rename(columns={"team_a": "t"})]).sort_values("date")
    seq["n"] = seq.groupby(["season", "t"]).cumcount()
    gn = seq.set_index(["game_id", "t"]).n
    nh = gn.reindex(list(zip(g.game_id, g.team))).values; na = gn.reindex(list(zip(g.game_id, g.team_a))).values
    def feats(season, teams):
        idx = list(zip(season, teams.astype(int)))
        return np.nan_to_num(Z.reindex(idx).values)
    Fh, Fa = feats(g.season, g.team), feats(g.season, g.team_a)
    C = len(E.COMP); nf = len(RF)
    G = np.zeros((2 * C, nf))   # rows 0..C-1 offense comps, C..2C-1 defense comps
    lam = 20.0
    for k, c in enumerate(E.COMP):
        ys, xs, ws = [], [], []
        for suf, hp, Fo, Fd, no, nd in (("", "h_", Fh, Fa, nh, na), ("_a", "a_", Fa, Fh, na, nh)):
            y = E.logit(g["y_" + c + suf].values) - E.logit(p0[hp + c].values)
            n = g["n_" + c + suf].values
            # offense row uses the offense team's early games, defense row the defending team's early games
            wo = (no < NG); wd = (nd < NG)
            ok = (g.season.values <= FIT_TO) & (g.season.values > g.season.min()) & (n > 0) & (wo | wd)
            x = np.hstack([Fo * wo[:, None], Fd * wd[:, None]])
            ys.append(y[ok]); xs.append(x[ok]); ws.append(np.minimum(n[ok], 200))
        y, x, w = np.concatenate(ys), np.vstack(xs), np.concatenate(ws)
        y = y - np.average(y, weights=w)
        A = (x * w[:, None]).T @ x + lam * w.mean() * np.eye(x.shape[1])
        b = np.linalg.solve(A, (x * w[:, None]).T @ y)
        G[k], G[C + k] = b[:nf], b[nf:]
    # pace: log-poss residual on both teams' features
    yp = np.log(g.poss.values) - np.log(p0.poss_hat.values)
    okp = (g.season.values <= FIT_TO) & (g.season.values > g.season.min()) & ((nh < NG) | (na < NG))
    xp = Fh * (nh < NG)[:, None] + Fa * (na < NG)[:, None]
    GP = np.linalg.solve(xp[okp].T @ xp[okp] + lam * np.eye(nf), xp[okp].T @ (yp[okp] - yp[okp].mean()))
    if verbose: print(pd.DataFrame(np.vstack([G, GP]), index=[f"O_{c}" for c in E.COMP] + [f"D_{c}" for c in E.COMP] + ["pace"], columns=RF).round(4).to_string())

    def start(R, season):
        if season == g.season.min(): return
        for t, i in R.idx.items():
            if (season, int(t)) not in Z.index: continue
            f = np.nan_to_num(Z.loc[(season, int(t))].values)
            R.O[i] += SCALE * G[:C] @ f; R.D[i] += SCALE * G[C:] @ f; R.Pace[i] += SCALE * GP @ f

    return start

if __name__ == "__main__":
    prm = {**E.DEFAULT, **json.load(open(sys.argv[1]))}
    NG = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    SCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    tg = pd.read_parquet(f"{HERE}/data/teamgames_cbb.parquet")
    p0, _ = E.run(tg, prm)
    start = prior_hook(tg, prm, p0, NG, SCALE, verbose=True)
    p1, _ = E.run(tg, prm, hooks={"season_start": start})

    base = pd.read_parquet(f"{HERE}/data/baseline_cbb.parquet")
    keep = lambda d: d[(d.season > FIT_TO) & d.game_id.isin(base.game_id)]
    A, B = keep(p0), keep(p1)
    bt = base.set_index("game_id").loc[B.game_id].reset_index()
    print("NG", NG, "SCALE", SCALE)
    print(pd.DataFrame({"today": H.score(bt), "frame": H.score(A), "frame+prior": H.score(B)}).loc[["n", "team_mae", "team_bias", "margin_mae", "total_mae"]].to_string())
    early = (B.groupby("season").date.rank(method="first") / B.groupby("season").date.transform("size") < 0.2).values
    print("first 20% of season margin MAE: frame", round(H.score(A[early])["margin_mae"], 3), "prior", round(H.score(B[early])["margin_mae"], 3), "today", round(H.score(bt[early])["margin_mae"], 3))
    out = {}
    for loss in ("team_abs", "margin_abs", "total_abs"):
        out[loss] = [H.compare(A, B, loss), H.compare(bt, B, loss)]; print("vs frame", out[loss][0]); print("vs today", out[loss][1])
    B.to_parquet(f"{HERE}/data/prior_cbb_n{NG}_s{SCALE}.parquet"); json.dump(out, open(f"{HERE}/data/m_prior_n{NG}_s{SCALE}.json", "w"), default=float)

