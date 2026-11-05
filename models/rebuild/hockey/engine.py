"""Hockey engine (NHL), layer 1 + expected goals.

Layer 1, ratings: each team has an offense and a defense rating (log scale) for every scoring component, tracked
by a Kalman filter walking forward one date at a time; each goalie has his own rating.
Components:
  evx  even-strength expected goals per 60 (shot volume x shot quality)
  ppt  power-play minutes per game (offense = penalties drawn, defense = penalties taken)
  ppx  power-play expected goals per 60 (offense = power play, defense = penalty kill)
  fin  goals per expected goal for the team's shooters (finishing; mostly luck, so it moves slowly)
  goalie  goals allowed per expected goal faced, per goalie (save skill; also mostly luck)
Expected goals for a team = EV xG rate x EV minutes x finishing x opposing goalie
  + PP minutes x PP xG rate x finishing x opposing goalie + short-handed and empty-net goals (league rates)
  + its share of overtime / shootout goals.
"""
import numpy as np, pandas as pd

COMP = ["evx", "ppt", "ppx", "fin"]
NBAR = np.array([34.0, 4.8, 8.0, 3.6])   # typical effective counts per team-game

DEFAULT = dict(q=0.00002, p0=0.02, carry=0.6, q_off=0.01, tau=(0.08, 0.25, 0.2, 0.1),
               fin_scale=0.3,          # finishing ratings move this much slower (luck)
               g_p0=0.01, g_q=0.00001, g_tau=0.15, g_carry=0.7, g_scale=1.0,
               hca_gain=0.002, mu_gain=0.01)

def obs(t):
    """Observed ratio to exposure (count / exposure, so the update is mean-unbiased) and effective counts,
    per team-game (offense view). The rating update uses y/expected - 1 as the log-scale step."""
    ev_h = t.toi_ev.clip(lower=600) / 3600
    o = pd.DataFrame({
        "evx": t.xg_ev / ev_h,
        "ppt": t.toi_pp / 60,
        "ppx": t.xg_pp / (t.toi_pp.clip(lower=1) / 3600),
        "fin": (t.g_ev + t.g_pp) / (t.xg_ev + t.xg_pp).clip(lower=0.05),
    })
    # information counts from exposure, never from the outcome (outcome-based weights bias the level upward)
    n = pd.DataFrame({"evx": 41.0 * ev_h, "ppt": pd.Series(4.8, index=t.index), "ppx": 85.0 * t.toi_pp / 3600 + 0.5,
                      "fin": (t.xg_ev + t.xg_pp).clip(lower=0.2)})
    return o, n

class Ratings:
    def __init__(self, teams, prm, mu0):
        self.prm = prm; self.idx = {t: i for i, t in enumerate(teams)}; T, C = len(teams), len(COMP)
        self.O = np.zeros((T, C)); self.D = np.zeros((T, C))
        sc = np.array([1, 1, 1, prm["fin_scale"]])
        self.sc = sc
        self.PO = np.full((T, C), prm["p0"]) * sc; self.PD = np.full((T, C), prm["p0"]) * sc
        self.PD[:, 3] = 0.0   # finishing is the shooters' own; the defense side of finishing is the goalie
        self.mu = np.array(mu0, float); self.hca = np.zeros(C)
        self.G = {}; self.PG = {}; self.gn = {}; self.last = np.full(T, np.nan); self.glast = {}

    def age(self, i, day):
        if np.isnan(self.last[i]): return
        dd = day - self.last[i]
        if dd > 0:
            self.PO[i] += self.prm["q"] * dd * self.sc; self.PD[i, :3] += self.prm["q"] * dd

    def recenter(self):
        """Keep team ratings centred on zero so the league level lives in mu only (otherwise shrinking the
        ratings at a new season also shifts the league level)."""
        act = ~np.isnan(self.last)
        if act.sum() < 2: return
        mo, md = self.O[act].mean(0), self.D[act].mean(0)
        self.mu += mo + md; self.O[act] -= mo; self.D[act] -= md

    def new_season(self):
        c, Q = self.prm["carry"], self.prm["q_off"]
        self.recenter()
        self.O *= c; self.D *= c
        self.PO = np.minimum(self.PO * c * c + Q * self.sc, (self.prm["p0"] + Q) * self.sc)
        self.PD = np.minimum(self.PD * c * c + Q, self.prm["p0"] + Q); self.PD[:, 3] = 0
        if self.prm.get("g_center") and self.gn:   # usage-weighted average goalie = 0; the level moves to finishing
            w = np.array([self.gn[k] for k in self.gn]); m = np.dot(w, [self.G[k] for k in self.gn]) / w.sum()
            for k in self.G: self.G[k] -= m
            self.mu[3] += m; self.gn = {}
        gc = self.prm["g_carry"]
        for k in self.G: self.G[k] *= gc; self.PG[k] = self.PG[k] * gc * gc + self.prm["g_q"] * 60

    def goalie(self, g):
        if g is None or g != g: return 0.0, 0.0
        if g not in self.G: self.G[g] = self.prm.get("g_new", 0.0); self.PG[g] = self.prm["g_p0"]
        self.gn[g] = self.gn.get(g, 0) + 1
        return self.G[g], self.PG[g]

    def predict(self, h, a, neutral, gk_a=None, gk_h=None):
        """Log rates for each side's offense, plus the opposing goalie's rating."""
        i, j = self.idx[h], self.idx[a]; s = 0.0 if neutral else 1.0
        lh = self.mu + self.O[i] + self.D[j] + s * self.hca
        la = self.mu + self.O[j] + self.D[i] - s * self.hca
        return lh, la, self.goalie(gk_a)[0], self.goalie(gk_h)[0]

    def update(self, off, de, side, y, n, gk):
        prm = self.prm; i, j = self.idx[off], self.idx[de]
        pred = self.mu + self.O[i] + self.D[j] + side * self.hca
        gv, gp = self.goalie(gk)
        pred = pred.copy(); pred[3] += gv
        R = 1.0 / np.maximum(n, 1e-3) + np.array(prm["tau"]) ** 2
        v = np.clip(y / np.exp(pred) - 1, -1.5, 3.0)   # Poisson-style step: mean-unbiased on the rate scale
        v = np.where(np.isfinite(v), v, 0.0)
        S = self.PO[i] + self.PD[j] + R
        S3 = S[3] + gp * prm["g_scale"]
        ko, kd = self.PO[i] / S, self.PD[j] / S
        ko[3] = self.PO[i, 3] / S3
        self.O[i] += ko * v; self.D[j] += kd * v
        self.PO[i] *= (1 - ko); self.PD[j] *= (1 - kd)
        if gk is not None and gk == gk:
            kg = gp * prm["g_scale"] / S3
            self.G[gk] += kg * v[3]; self.PG[gk] = gp * (1 - kg)
        w = np.minimum(n / NBAR, 3.0)   # weight league-level steps by information, so the league mean is a ratio of sums
        self.mu += prm["mu_gain"] * w * v / 2
        if side: self.hca += prm["hca_gain"] * side * w * v

def games(tg):
    tg = tg.reset_index(drop=True)
    o, n = obs(tg)
    tg = pd.concat([tg, o.add_prefix("y_"), n.add_prefix("n_")], axis=1)
    h = tg[tg.ha == "home"].set_index("game_id"); a = tg[tg.ha == "away"].set_index("game_id")
    g = h.join(a, rsuffix="_a", how="inner").reset_index()
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)

def league(tg):
    """League constants by season: short-handed and empty-net goals per team-game, overtime goals, EV minutes."""
    s = tg.groupby("season").agg(sh=("g_sh", "mean"), en=("g_en", "mean"), ev_min=("toi_ev", lambda x: x.mean() / 60),
                                 otg=("so", "mean"))
    return s

def run(tg, prm=None, hooks=None):
    prm = {**DEFAULT, **(prm or {})}
    g = games(tg)
    teams = pd.unique(pd.concat([g.team, g.team_a]))
    first = g[g.season == g.season.min()]
    mu0 = [np.log(first["y_" + c].replace([np.inf, -np.inf], np.nan).mean()) for c in COMP]
    R = Ratings(teams, prm, mu0)
    L = league(tg)
    rows = []; cur = None; day0 = g.date.min(); hist = {}; usual = {}
    lg_sh, lg_en, lg_ev = 0.075, 0.16, 49.0
    so_rate = 0.07     # share of games to a shootout: the winner's +1 (OT goals are already in the even/PP rates)
    for d, gd in g.groupby("date", sort=True):
        day = (d - day0).days
        if gd.season.iloc[0] != cur:
            if cur is not None:
                R.new_season()
                lg_sh, lg_en, lg_ev, so_rate = L.loc[cur, "sh"], L.loc[cur, "en"], L.loc[cur, "ev_min"], L.loc[cur, "otg"]
            cur = gd.season.iloc[0]
        recs = gd.to_dict("records")
        for r in recs:
            R.age(R.idx[r["team"]], day); R.age(R.idx[r["team_a"]], day)
            gk_a, gk_h = r.get("goalie"), r.get("goalie_a")   # away goalie (faced by home shooters), home goalie
            mode = prm.get("gk_mode", "actual")
            if mode == "none": gk_a = gk_h = None
            elif mode == "usual":   # the team's most-used goalie over its last 10 games (no starter news)
                gk_a, gk_h = usual.get(r["team_a"]), usual.get(r["team"])
            lh, la, gka, gkh = R.predict(r["team"], r["team_a"], False, gk_a, gk_h)
            out = {"game_id": r["game_id"], "season": r["season"], "date": d, "home": r["team"], "away": r["team_a"],
                   "playoff": bool(r["playoff"]), "act_h": r["gf"], "act_a": r["gf_a"]}
            for side, l, gk in (("h", lh, gka), ("a", la, gkh)):
                evx, ppt, ppx, fin = np.exp(l[0]), np.exp(l[1]), np.exp(l[2]), np.exp(l[3] + gk)
                out[f"evg_{side}"] = evx * lg_ev / 60 * fin
                out[f"ppg_{side}"] = ppt / 60 * ppx * fin
                out[f"ppmin_{side}"] = ppt
            for side in ("h", "a"):
                out[f"pred_{side}"] = out[f"evg_{side}"] + out[f"ppg_{side}"] + lg_sh + lg_en + so_rate * 0.5
            if hooks and "predict" in hooks: hooks["predict"](R, r, out)
            rows.append(out)
        for r in recs:
            yh = np.array([r["y_" + c] for c in COMP]); nh = np.array([r["n_" + c] for c in COMP])
            ya = np.array([r["y_" + c + "_a"] for c in COMP]); na = np.array([r["n_" + c + "_a"] for c in COMP])
            R.update(r["team"], r["team_a"], 1, yh, nh, r.get("goalie"))
            R.update(r["team_a"], r["team"], -1, ya, na, r.get("goalie_a"))
            R.last[R.idx[r["team"]]] = day; R.last[R.idx[r["team_a"]]] = day
            for tm, gk in ((r["team_a"], r.get("goalie")), (r["team"], r.get("goalie_a"))):
                if gk is None or gk != gk: continue
                hist.setdefault(tm, []).append(gk); hist[tm] = hist[tm][-10:]
                usual[tm] = max(set(hist[tm]), key=hist[tm].count)
    return pd.DataFrame(rows), R
