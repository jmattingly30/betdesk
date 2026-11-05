"""Layer 3 context modules for the basketball engine, tested one at a time on top of what has been kept.
Each module gives, per game and side, a feature block (own team, then opponent). Ridge weights per possession
component (fit on seasons <= FIT_TO only) turn features into logit shifts inside the engine.
Usage: python3 ctx.py sport params.json kept_modules new_module   e.g. ctx.py nba data/tuned_nba.json lineup schedule
Modules: lineup (NBA), schedule, pace (not a rate module: handled by engine)."""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import harness as H, engine as E

def lineup_block(tg, g, HL=15):
    L = pd.read_parquet(os.environ.get("LINEUP_AGG", f"{HERE}/../../nba-team/data/lineup_agg.parquet"))
    def feats(suf):
        q = lambda c: L[f"r_{c}{suf}"]
        return pd.DataFrame({"pts": q("pts"), "fta": q("fta"), "s3": q("fg3a") / q("fga"), "p3": q("fg3m") / q("fg3a"),
                             "tov": q("tov"), "oreb": q("oreb"), "ast": q("ast"), "dreb": q("dreb"), "stl": q("stl"),
                             "blk": q("blk"), "pf": q("pf"), "rapm": q("rapm")})
    FA, FK, FF = feats(""), feats("_known"), feats("_full")
    for c in ("pts", "rapm"):
        FA["lost_" + c] = FF[c] - FA[c]; FK["lost_" + c] = FF[c] - FK[c]
    F = list(FA.columns)
    L = pd.concat([L[["game_id", "team"]], FA.add_prefix("a_"), FK.add_prefix("k_")], axis=1)
    L = L.merge(tg[["game_id", "team", "date"]], on=["game_id", "team"]).sort_values(["team", "date"])
    alpha = 1 - 0.5 ** (1 / HL)
    rel = [c for c in F if not c.startswith("lost_")]
    base = L.groupby("team")[[f"a_{c}" for c in rel]].transform(lambda s: s.shift(1).ewm(alpha=alpha, ignore_na=True).mean())
    for c in F:
        if c in rel:
            b = base[f"a_{c}"].fillna(L[f"a_{c}"]); L[f"k_{c}"] = L[f"k_{c}"] - b; L[f"a_{c}"] = L[f"a_{c}"] - b
    L = L.fillna(0.0)
    for c in F:
        s = L[f"a_{c}"].std(); L[f"k_{c}"] /= s; L[f"a_{c}"] /= s
    LI = L.set_index(["game_id", "team"])
    def side(own, opp, kind):
        a = LI.reindex(list(zip(g.game_id, own)))[[f"{kind}_{c}" for c in F]].values
        b = LI.reindex(list(zip(g.game_id, opp)))[[f"{kind}_{c}" for c in F]].values
        return np.nan_to_num(np.hstack([a, b]))
    names = [f"own_{c}" for c in F] + [f"opp_{c}" for c in F]
    # predict with the lineup known before the game; fit weights on who actually played
    return names, side(g.team, g.team_a, "k"), side(g.team_a, g.team, "k"), side(g.team, g.team_a, "a"), side(g.team_a, g.team, "a")

def schedule_block(tg, g):
    t = tg[["game_id", "team", "date", "season"]].sort_values(["team", "date"]).copy()
    d = t.groupby("team").date.diff().dt.days
    t["b2b"] = (d == 1).astype(float)
    t["rest3"] = (d >= 3).astype(float) * (d < 30)
    t["first"] = d.isna() | (d >= 30)
    # games in the last 4 and 7 days (density)
    t["g4"] = 0.0; t["g7"] = 0.0
    for team, s in t.groupby("team"):
        dd = s.date.values.astype("datetime64[D]").astype(int)
        t.loc[s.index, "g4"] = [((dd[:i] >= x - 4)).sum() for i, x in enumerate(dd)]
        t.loc[s.index, "g7"] = [((dd[:i] >= x - 7)).sum() for i, x in enumerate(dd)]
    # road-trip length: consecutive away games so far
    ha = tg.set_index(["game_id", "team"]).ha
    t["away"] = [ha.get((gi, tm)) == "away" for gi, tm in zip(t.game_id, t.team)]
    t["trip"] = t.groupby("team").away.transform(lambda s: s.groupby((~s).cumsum()).cumsum()).astype(float)
    t["trip5"] = (t.trip >= 5).astype(float)
    F = os.environ.get("SCHED_F", "b2b,rest3,trip5").split(",")
    TI = t.set_index(["game_id", "team"])[F].astype(float)
    def side(own, opp):
        a = TI.reindex(list(zip(g.game_id, own))).values; b = TI.reindex(list(zip(g.game_id, opp))).values
        return np.nan_to_num(np.hstack([a, b]))
    names = [f"own_{c}" for c in F] + [f"opp_{c}" for c in F]
    h, a = side(g.team, g.team_a), side(g.team_a, g.team)
    return names, h, a, h, a

def avail_block(tg, g):
    """CBB: value of regulars missing (models/cbb prep_availability). Predict with continuing absences (known
    before the game), fit on all absences."""
    A = pd.read_parquet(f"{HERE}/../../cbb/data/cbb_availability.parquet").drop_duplicates(["game_id", "team"])
    A["team"] = A.team.astype(int); AI = A.set_index(["game_id", "team"])
    def col(c, teams): return np.nan_to_num(AI[c].reindex(list(zip(g.game_id, teams.astype(int)))).values)
    def side(c, own, opp): return np.c_[col(c, own), col(c, opp)]
    names = ["own_abs", "opp_abs"]
    return (names, side("abs_val_cont", g.team, g.team_a), side("abs_val_cont", g.team_a, g.team),
            side("abs_val", g.team, g.team_a), side("abs_val", g.team_a, g.team))

def venue_block(tg, g):
    """NBA altitude: playing at Denver / Utah (home or away side). Own = this side's offense at altitude."""
    alt = g.team.isin(["DEN", "UTAH"]).values & ~g.neutral.values
    vis = alt.astype(float)   # game is at altitude
    h = np.c_[vis * 0, vis]   # home side offense: [visitor flag, at-altitude-as-home]
    a = np.c_[vis, vis * 0]   # away side offense: visitor at altitude
    names = ["visitor_at_altitude", "home_at_altitude"]
    return names, h, a, h, a

BLOCKS = {"venue": venue_block, "lineup": lineup_block, "schedule": schedule_block, "avail": avail_block}

def build(tg, g, mods):
    names, out = [], [[], [], [], []]
    for m in mods:
        n, *xs = BLOCKS[m](tg, g); names += [f"{m}:{x}" for x in n]
        for o, x in zip(out, xs): o.append(x)
    return names, *[np.hstack(o) for o in out]

def fit(g, p, Xh, Xa, fit_to, lam=50.0):
    m = (g.season <= fit_to).values
    B = np.zeros((Xh.shape[1], len(E.COMP)))
    for k, c in enumerate(E.COMP):
        ys, xs, ws = [], [], []
        for suf, X, hp in (("", Xh, "h_"), ("_a", Xa, "a_")):
            y = E.logit(g["y_" + c + suf].values) - E.logit(p[hp + c].values)
            ok = m & (g["n_" + c + suf].values > 0)
            ys.append(y[ok]); xs.append(X[ok]); ws.append(np.minimum(g["n_" + c + suf].values[ok], 200))
        y, x, w = np.concatenate(ys), np.vstack(xs), np.concatenate(ws)
        mu_x = np.average(x, axis=0, weights=w); sx = x.std(0) + 1e-9
        z = (x - mu_x) / sx; y = y - np.average(y, weights=w)
        A = (z * w[:, None]).T @ z + lam * w.mean() * np.eye(z.shape[1])
        B[:, k] = np.linalg.solve(A, (z * w[:, None]).T @ y) / sx
    return B

def run_with(tg, prm, g, Xh, Xa, B, hooks=None):
    if Xh.shape[1] == 0: return E.run(tg, prm, hooks=hooks)[0]
    ph, pa = Xh @ B, Xa @ B
    sh = {gid: (ph[i], pa[i], 0 * ph[i], 0 * pa[i]) for i, gid in enumerate(g.game_id)}
    return E.run(tg, prm, shifts=sh, hooks=hooks)[0]

def model(tg, prm, mods, fit_to):
    """Rating modules (prior) change layer 1 and are applied first; the rest are layer-3 rate shifts."""
    g = E.games_from_teamgames(tg)
    hooks = None
    if "prior" in mods:
        import m_prior
        hooks = {"season_start": m_prior.prior_hook(tg, prm, FIT_TO=fit_to)}
        mods = [m for m in mods if m != "prior"]
    p0 = E.run(tg, prm, hooks=hooks)[0]
    if not mods: return p0, None, []
    names, Xh, Xa, Fh, Fa = build(tg, g, mods)
    m = (g.season <= fit_to).values   # center on the fit features (fit seasons) so a shift has no built-in intercept
    mu = np.vstack([Fh[m], Fa[m]]).mean(0); Xh, Xa, Fh, Fa = Xh - mu, Xa - mu, Fh - mu, Fa - mu
    B = fit(g, p0, Fh, Fa, fit_to)   # one pass on the frame's residuals
    return run_with(tg, prm, g, Xh, Xa, B, hooks), B, names

if __name__ == "__main__":
    sport, pf = sys.argv[1], sys.argv[2]
    kept = [m for m in sys.argv[3].split(",") if m] if len(sys.argv) > 3 else []
    new = sys.argv[4]
    prm = {**E.DEFAULT, **json.load(open(pf))}
    fit_to = 2017 if sport == "nba" else 2016
    tg = pd.read_parquet(f"{HERE}/data/teamgames_{sport}.parquet")
    pa, _, _ = model(tg, prm, kept, fit_to)
    pb, B, names = model(tg, prm, kept + [new], fit_to)
    base = pd.read_parquet(f"{HERE}/data/baseline_{'nba_known' if sport == 'nba' else sport}.parquet")
    keep = lambda d: d[(d.season > fit_to) & d.game_id.isin(base.game_id)]
    A, Bp = keep(pa), keep(pb)
    print(pd.DataFrame(B, index=names, columns=E.COMP).round(4).to_string())
    print(pd.DataFrame({"before": H.score(A), "with " + new: H.score(Bp)}).loc[["n", "team_mae", "team_bias", "margin_mae", "total_mae"]].to_string())
    out = {}
    for loss in ("team_abs", "margin_abs", "total_abs"):
        r = H.compare(A, Bp, loss); out[loss] = r; print(r)
    pb.to_parquet(f"{HERE}/data/ctx_{sport}_{'_'.join(kept + [new])}.parquet")
    json.dump(out, open(f"{HERE}/data/ctx_{sport}_{'_'.join(kept + [new])}.json", "w"), default=float)

def load_tg(sport):
    """Team-games plus (CBB) each team's conference that season, for the conference layer."""
    tg = pd.read_parquet(f"{HERE}/data/teamgames_{sport}.parquet")
    if sport != "cbb": return tg, 0
    G = pd.read_parquet(f"{HERE}/../../cbb/data/cbb_games.parquet")[["game_id", "home", "away", "home_conference_id", "away_conference_id"]]
    cf = pd.concat([G[["game_id", "home", "home_conference_id"]].set_axis(["game_id", "team", "c"], axis=1),
                    G[["game_id", "away", "away_conference_id"]].set_axis(["game_id", "team", "c"], axis=1)]).dropna()
    cf["team"] = cf.team.astype(int); cf["game_id"] = cf.game_id.astype(tg.game_id.dtype)
    codes = {c: i for i, c in enumerate(sorted(cf.c.unique()))}; cf["conf"] = cf.c.map(codes)
    tg = tg.merge(cf[["game_id", "team", "conf"]].drop_duplicates(["game_id", "team"]), on=["game_id", "team"], how="left")
    tg["conf"] = tg.conf.fillna(-1).astype(int)
    return tg, len(codes)

# the kept stack per sport: (modules, params file); see ledger.csv for why each piece is in
STACK = {"nba": (["lineup", "schedule"], "data/tuned_stack_nba.json"),
         "cbb": (["prior", "avail"], "data/stack_params_cbb.json")}

def stack_params(sport, nconf):
    prm = {**E.DEFAULT, **json.load(open(f"{HERE}/{STACK[sport][1]}"))}
    if sport == "cbb": prm["n_conf"] = nconf
    return prm
