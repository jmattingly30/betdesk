"""Basketball engine (CBB + NBA), layer 1 + layer 2.

Layer 1, ratings: every team has an offense and a defense rating for each possession component, in logit space,
tracked by a Kalman filter that walks forward one date at a time (uncertainty grows between games and over the
offseason, shrinks with each game, weighted by how many attempts the game had).
Components (offense view; the defense rating is what the team allows):
  tov  turnovers per possession          fta  free-throw trips per scoring attempt
  s3   3-point share of FGA               p3   3P%        p2  2P%
  ftp  FT%                               orb  offensive rebounds per missed FG
  pace possessions (log scale; offense = own pace, defense = opponent's pace effect)
Layer 2, game engine: possession math gives expected points; sim.py plays the possessions out.
"""
import numpy as np, pandas as pd

COMP = ["tov", "fta", "s3", "p3", "p2", "ftp", "orb"]

def components(t):
    """Observed component rates and their trial counts for each team-game (offense view)."""
    trips = 0.44 * t.fta
    att = t.fga + trips
    miss = (t.fga - t.fgm).clip(lower=1)
    obs = pd.DataFrame({
        "tov": t.tov / t.poss, "fta": trips / att, "s3": t.tpa / t.fga.clip(lower=1),
        "p3": t.tpm / t.tpa.clip(lower=1), "p2": (t.fgm - t.tpm) / (t.fga - t.tpa).clip(lower=1),
        "ftp": t.ftm / t.fta.clip(lower=1), "orb": t.orb / miss})
    n = pd.DataFrame({"tov": t.poss, "fta": att, "s3": t.fga, "p3": t.tpa, "p2": t.fga - t.tpa,
                      "ftp": t.fta, "orb": miss})
    return obs, n

def logit(p): p = np.clip(p, 1e-3, 1 - 1e-3); return np.log(p / (1 - p))
def expit(x): return 1 / (1 + np.exp(-x))

def ppp(r):
    """Expected points per possession from component rates (dict of arrays). Possession identity:
    poss = FGA - ORB + TOV + 0.44 FTA holds exactly under this process."""
    t, f, s3, p3, p2, ftp, orb = (r[c] for c in COMP)
    pts_att = f * (ftp / 0.44) + (1 - f) * (s3 * 3 * p3 + (1 - s3) * 2 * p2)
    miss = (1 - f) * (s3 * (1 - p3) + (1 - s3) * (1 - p2))
    return (1 - t) * pts_att / (1 - miss * orb)

DEFAULT = dict(
    q=0.0004,          # rating drift per day (variance, logit units) -> recency
    p0=0.04,           # starting uncertainty of a team rating
    carry=0.6,         # share of last season's rating kept into the new season
    q_off=0.03,        # extra uncertainty added in the offseason
    tau=0.0,           # game-to-game wobble beyond sampling noise (logit sd)
    d_scale=None,      # per-component multiplier on defense uncertainty (luck module); None = 1
    hca_gain=0.002, mu_gain=0.01,
    pace_q=0.0002, pace_p0=0.003, pace_tau=0.04,
)

class Ratings:
    def __init__(self, teams, prm, mu0, pace0):
        self.prm = prm; self.idx = {t: i for i, t in enumerate(teams)}; T = len(teams); C = len(COMP)
        self.O = np.zeros((T, C)); self.D = np.zeros((T, C))
        self.PO = np.full((T, C), prm["p0"]); self.PD = np.full((T, C), prm["p0"])
        ds = prm.get("d_scale") or {}
        self.dsc = np.array([ds.get(c, 1.0) for c in COMP]); self.PD *= self.dsc
        self.mu = logit(np.array(mu0)); self.hca = np.zeros(C)
        self.Pace = np.zeros(T); self.PP = np.full(T, prm["pace_p0"]); self.mu_pace = np.log(pace0); self.hca_pace = 0.0
        self.last = np.full(T, np.nan)
        # conference layer (CBB): a shared offense/defense rating per conference, so results between conferences move
        # every member; last row is "no conference" and never moves
        nc = prm.get("n_conf", 0); self.nc = nc
        self.CO = np.zeros((nc + 1, C)); self.CD = np.zeros((nc + 1, C))
        self.PCO = np.full((nc + 1, C), prm.get("pc0", 0.0)); self.PCD = self.PCO.copy()
        cm = prm.get("conf_comp")   # components the conference layer covers (None = all)
        if cm is not None: self.PCO[:, [k for k, c in enumerate(COMP) if c not in cm]] = 0; self.PCD[:] = self.PCO
        self.cmask = np.array([1.0 if cm is None or c in cm else 0.0 for c in COMP])
        self.PCO[nc] = 0; self.PCD[nc] = 0

    def age(self, i, day):
        if np.isnan(self.last[i]): return
        dd = day - self.last[i]
        if dd <= 0: return
        self.PO[i] += self.prm["q"] * dd; self.PD[i] += self.prm["q"] * dd * self.dsc; self.PP[i] += self.prm["pace_q"] * dd

    def boost(self):
        """Early-season league level: learn the new season's scoring level faster for the first mu_boost_n games."""
        return 1.0 + self.prm.get("mu_boost", 0.0) * (getattr(self, "ng", 0) < self.prm.get("mu_boost_n", 0))

    def new_season(self):
        c, Q = self.prm["carry"], self.prm["q_off"]
        self.ng = 0
        if self.prm.get("recenter"):   # league level lives in mu only (else shrinking ratings shifts the level)
            act = ~np.isnan(self.last)
            mo, md = self.O[act].mean(0), self.D[act].mean(0); self.mu += mo + md; self.O[act] -= mo; self.D[act] -= md
            pm = self.Pace[act].mean(); self.mu_pace += 2 * pm; self.Pace[act] -= pm
        self.O *= c; self.D *= c; self.Pace *= c
        self.PO = np.minimum(self.PO * c * c + Q, self.prm["p0"] + Q)
        self.PD = np.minimum(self.PD * c * c + Q * self.dsc, (self.prm["p0"] + Q) * self.dsc)
        self.PP = self.PP * c * c + self.prm["pace_p0"]
        if self.nc:
            cc, Qc = self.prm.get("carry_c", 0.9), self.prm.get("qc_off", 0.002)
            self.CO *= cc; self.CD *= cc
            self.PCO = self.PCO * cc * cc + Qc * self.cmask; self.PCD = self.PCD * cc * cc + Qc * self.cmask
            self.PCO[self.nc] = 0; self.PCD[self.nc] = 0

    def cidx(self, c):
        c = int(c) if c == c else -1
        return c if 0 <= c < self.nc else self.nc

    def predict(self, h, a, neutral, ch=-1, ca=-1):
        i, j = self.idx[h], self.idx[a]; s = 0.0 if neutral else 1.0
        lh = self.mu + self.O[i] + self.D[j] + s * self.hca + self.CO[ch] + self.CD[ca]
        la = self.mu + self.O[j] + self.D[i] - s * self.hca + self.CO[ca] + self.CD[ch]
        poss = np.exp(self.mu_pace + self.Pace[i] + self.Pace[j] + s * 0.0)
        return lh, la, poss, (self.PO[i] + self.PD[j]), (self.PO[j] + self.PD[i])

    def update(self, off, de, side, y, n, poss_obs, shift=0.0, co=-1, cd=-1):
        """side: +1 home offense, -1 away offense, 0 neutral. shift: known context (e.g. who actually played),
        so the ratings learn strength net of it."""
        prm = self.prm; i, j = self.idx[off], self.idx[de]
        pred = self.mu + self.O[i] + self.D[j] + side * self.hca + shift + self.CO[co] + self.CD[cd]
        p = expit(pred)
        R = 1.0 / np.maximum(n * p * (1 - p), 1e-6) + prm["tau"] ** 2
        v = logit(y) - pred
        ok = n > 0
        inter = self.nc and co != cd
        S = self.PO[i] + self.PD[j] + R + ((self.PCO[co] + self.PCD[cd]) if inter else 0.0)
        if inter:   # conference ratings learn only from games between conferences
            kc, kdc = self.PCO[co] / S, self.PCD[cd] / S
            self.CO[co] += np.where(ok, kc * v, 0); self.CD[cd] += np.where(ok, kdc * v, 0)
            self.PCO[co] = np.where(ok, self.PCO[co] * (1 - kc), self.PCO[co]); self.PCD[cd] = np.where(ok, self.PCD[cd] * (1 - kdc), self.PCD[cd])
            self.PCO[self.nc] = 0; self.PCD[self.nc] = 0
        ko, kd = self.PO[i] / S, self.PD[j] / S
        self.O[i] += np.where(ok, ko * v, 0); self.D[j] += np.where(ok, kd * v, 0)
        self.PO[i] = np.where(ok, self.PO[i] * (1 - ko), self.PO[i]); self.PD[j] = np.where(ok, self.PD[j] * (1 - kd), self.PD[j])
        self.mu += prm["mu_gain"] * self.boost() * np.where(ok, v, 0) / 2
        if side != 0: self.hca += prm["hca_gain"] * side * np.where(ok, v, 0)
        return v

    def update_pace(self, h, a, poss, neutral):
        prm = self.prm; i, j = self.idx[h], self.idx[a]
        pred = self.mu_pace + self.Pace[i] + self.Pace[j]
        v = np.log(poss) - pred; S = self.PP[i] + self.PP[j] + prm["pace_tau"] ** 2
        ki, kj = self.PP[i] / S, self.PP[j] / S
        self.Pace[i] += ki * v; self.Pace[j] += kj * v
        self.PP[i] *= (1 - ki); self.PP[j] *= (1 - kj)
        self.mu_pace += prm["mu_gain"] * self.boost() * v / 2
        self.ng = getattr(self, "ng", 0) + 1

def games_from_teamgames(tg):
    """One row per game: home side first (for neutral games the first listed team is 'home')."""
    tg = tg.copy()
    obs, n = components(tg)
    tg = pd.concat([tg, obs.add_prefix("y_"), n.add_prefix("n_")], axis=1)
    order = tg.ha.map({"home": 0, "away": 1, "neutral": 0}).fillna(0)
    tg = tg.assign(_o=order).sort_values(["date", "game_id", "_o", "team"])
    first = tg.groupby("game_id").head(1).set_index("game_id")
    second = tg.groupby("game_id").tail(1).set_index("game_id")
    g = first.join(second, rsuffix="_a").reset_index()
    g = g[g.team != g.team_a]
    g["neutral"] = (g.ha == "neutral") | (g.neutral.fillna(False).astype(bool)) | (g.ha == g.ha_a)
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)

def run(tg, prm=None, hooks=None, shifts=None):
    """Walk forward through every date: predict that date's games from ratings before it, then update.
    shifts: optional {game_id: (pred_h, pred_a, obs_h, obs_a)} logit offsets per component (layer 3 context:
    pred = what is known before the game, obs = what turned out, used in the update).
    Returns per-game predictions (expected rates, possessions, points)."""
    Z = np.zeros(len(COMP))
    prm = {**DEFAULT, **(prm or {})}
    g = games_from_teamgames(tg)
    teams = pd.unique(pd.concat([g.team, g.team_a]))
    first = g[g.season == g.season.min()]
    mu0 = [np.average(first["y_" + c], weights=first["n_" + c]) for c in COMP]
    R = Ratings(teams, prm, mu0, first.poss.mean())
    rows = []; cur_season = None
    day0 = g.date.min()
    for d, gd in g.groupby("date", sort=True):
        day = (d - day0).days
        if gd.season.iloc[0] != cur_season:
            if cur_season is not None:
                if hooks and "season_end" in hooks: hooks["season_end"](R, cur_season)
                R.new_season()
            cur_season = gd.season.iloc[0]
            if hooks and "season_start" in hooks: hooks["season_start"](R, cur_season)
        recs = gd.to_dict("records")
        for r in recs:
            R.age(R.idx[r["team"]], day); R.age(R.idx[r["team_a"]], day)
            ch, ca = R.cidx(r.get("conf", -1)), R.cidx(r.get("conf_a", -1))
            lh, la, poss, vh, va = R.predict(r["team"], r["team_a"], r["neutral"], ch, ca)
            sp = shifts.get(r["game_id"]) if shifts else None
            if sp is not None: lh = lh + sp[0]; la = la + sp[1]
            out = {"game_id": r["game_id"], "season": r["season"], "date": d, "home": r["team"], "away": r["team_a"],
                   "neutral": r["neutral"], "poss_hat": poss, "act_h": r["pts"], "act_a": r["pts_a"], "poss": r["poss"],
                   "playoff": bool(r.get("playoff", False))}
            for k, c in enumerate(COMP):
                out["h_" + c] = expit(lh[k]); out["a_" + c] = expit(la[k])
                out["vh_" + c] = vh[k]; out["va_" + c] = va[k]
            if hooks and "predict" in hooks: hooks["predict"](R, r, out)
            rows.append(out)
        for r in recs:
            po_fix = prm.get("po_mu_freeze") and bool(r.get("playoff", False))
            if po_fix: mu_s, mp_s, hca_s = R.mu.copy(), R.mu_pace, R.hca.copy()   # playoff games don't move the league level
            yh = np.array([r["y_" + c] for c in COMP]); nh = np.array([r["n_" + c] for c in COMP])
            ya = np.array([r["y_" + c + "_a"] for c in COMP]); na = np.array([r["n_" + c + "_a"] for c in COMP])
            side = 0 if r["neutral"] else 1
            sp = shifts.get(r["game_id"]) if shifts else None
            ch, ca = R.cidx(r.get("conf", -1)), R.cidx(r.get("conf_a", -1))
            R.update(r["team"], r["team_a"], side, yh, nh, r["poss"], Z if sp is None else sp[2], ch, ca)
            R.update(r["team_a"], r["team"], -side, ya, na, r["poss"], Z if sp is None else sp[3], ca, ch)
            R.update_pace(r["team"], r["team_a"], r["poss"], r["neutral"])
            if po_fix: R.mu, R.mu_pace, R.hca = mu_s, mp_s, hca_s
            R.last[R.idx[r["team"]]] = day; R.last[R.idx[r["team_a"]]] = day
    p = pd.DataFrame(rows)
    rh = {c: p["h_" + c].values for c in COMP}; ra = {c: p["a_" + c].values for c in COMP}
    p["ppp_h"] = ppp(rh); p["ppp_a"] = ppp(ra)
    p["pred_h"] = p.poss_hat * p.ppp_h; p["pred_a"] = p.poss_hat * p.ppp_a
    return p, R
