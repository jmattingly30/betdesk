"""Football engine (NFL first), layer 1: drive-component ratings.

Each team has an offense and a defense rating for each component, tracked by a Kalman filter one week at a time:
  sr     success rate per scrimmage play
  epp    EPA per play
  xr     explosive plays per play (20+ yd passes, 10+ yd runs)
  tor    turnovers per play (mostly luck: moves slowly)
  start  average drive start (yards from the end zone; offense = own returns/defense, defense = opponent's)
  drv    drives per game (pace; both teams share it)
  fin    points per drive beyond what the components imply (red-zone finishing / kicking; mostly luck)
Expected points = drives x points-per-drive(components) + finishing + return/defensive points (league rate).
The map from components to points per drive is fit once on actual games (2007-2014), so it is structure, not a
prediction fit. Information counts come from exposure (plays, drives), never from outcomes.
"""
import numpy as np, pandas as pd

COMP = ["sr", "epp", "xr", "tor", "start", "drv", "fin"]
SIG = np.array([0.5, 1.4, 0.38, 0.14, 25.0, 1.6, 2.4])     # per-unit sd of one play / drive (so var = SIG^2 / n)
DEFAULT = dict(q=0.0, p0=1.0, carry=0.6, q_off=0.3, hca=None, hca_gain=0.0005, mu_gain=0.002,
               tau=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), slow=(1, 1, 1, 0.3, 0.5, 1, 0.2))
MAP = dict(const=3.770, sr=0.825, epp=3.631, xr=0.571, tor=5.344, start=-0.034)

def obs(T):
    o = pd.DataFrame({"sr": T.succ / T.plays, "epp": T.epa / T.plays, "xr": T.expl / T.plays, "tor": T.turnovers / T.plays,
                      "start": T.start_y100, "drv": T.drives})
    o["fin"] = T.pts_off / T.drives - ppd(o)
    n = pd.DataFrame({"sr": T.plays, "epp": T.plays, "xr": T.plays, "tor": T.plays, "start": T.drives,
                      "drv": pd.Series(1.0, index=T.index), "fin": T.drives})
    return o.fillna(o.mean()), n.fillna(1)

def ppd(o):
    return MAP["const"] + sum(MAP[k] * o[k] for k in ("sr", "epp", "xr", "tor", "start"))

class Ratings:
    def __init__(self, teams, prm, mu0):
        self.prm = prm; self.idx = {t: i for i, t in enumerate(teams)}; T, C = len(teams), len(COMP)
        self.sl = np.array(prm["slow"]); self.var0 = (SIG / np.sqrt(np.array([62, 62, 62, 62, 11.4, 1, 11.4]))) ** 2
        self.O = np.zeros((T, C)); self.D = np.zeros((T, C))
        self.PO = np.tile(prm["p0"] * self.var0 * self.sl, (T, 1)); self.PD = self.PO.copy()
        self.PD[:, 6] = 0.0          # finishing belongs to the offense
        self.mu = np.array(mu0, float); self.hca = np.zeros(C)
    def recenter(self):
        mo, md = self.O.mean(0), self.D.mean(0); self.mu += mo + md; self.O -= mo; self.D -= md
    def new_season(self):
        c, Q = self.prm["carry"], self.prm["q_off"]
        self.recenter(); self.O *= c; self.D *= c
        cap = (self.prm["p0"] + Q) * self.var0 * self.sl
        self.PO = np.minimum(self.PO * c * c + Q * self.var0 * self.sl, cap); self.PD = np.minimum(self.PD * c * c + Q * self.var0 * self.sl, cap)
        self.PD[:, 6] = 0.0
    def week(self):
        q = self.prm["q"] * self.var0 * self.sl; self.PO += q; self.PD += q; self.PD[:, 6] = 0.0
    def predict(self, h, a, neutral):
        i, j = self.idx[h], self.idx[a]; s = 0.0 if neutral else 1.0
        return self.mu + self.O[i] + self.D[j] + s * self.hca, self.mu + self.O[j] + self.D[i] - s * self.hca
    def update(self, off, de, side, y, n):
        i, j = self.idx[off], self.idx[de]
        pred = self.mu + self.O[i] + self.D[j] + side * self.hca
        R = SIG ** 2 / np.maximum(n, 1) + np.array(self.prm["tau"]) ** 2
        v = y - pred
        S = self.PO[i] + self.PD[j] + R
        ko, kd = self.PO[i] / S, self.PD[j] / S
        self.O[i] += ko * v; self.D[j] += kd * v; self.PO[i] *= 1 - ko; self.PD[j] *= 1 - kd
        self.mu += self.prm["mu_gain"] * v
        if side: self.hca += self.prm["hca_gain"] * side * v

def games(T):
    T = T.reset_index(drop=True); o, n = obs(T)
    T = pd.concat([T, o.add_prefix("y_"), n.add_prefix("n_")], axis=1)
    h = T[T.is_home == 1].set_index("game_id"); a = T[T.is_home == 0].set_index("game_id")
    g = h.join(a, rsuffix="_a", how="inner").reset_index()
    g["wk"] = g.season * 100 + g.week
    return g.sort_values(["date", "game_id"]).reset_index(drop=True)

def points(l, ret):
    o = dict(zip(COMP, l))
    return o["drv"] * (ppd(o) + o["fin"]) + ret

def run(T, prm=None, hooks=None):
    prm = {**DEFAULT, **(prm or {})}
    g = games(T)
    teams = pd.unique(pd.concat([g.team, g.team_a]))
    first = g[g.season == g.season.min()]
    mu0 = [first["y_" + c].mean() for c in COMP]
    R = Ratings(teams, prm, mu0); ret = 1.15; lvl = 0.0   # lvl: walk-forward level fix for drives x ppd covariance
    rows = []; cur = None
    for wk, gw in g.groupby("wk", sort=True):
        if gw.season.iloc[0] != cur:
            if cur is not None: R.new_season(); ret = g[g.season == cur].pts_ret.mean()
            cur = gw.season.iloc[0]
            if hooks and "season_start" in hooks: hooks["season_start"](R, cur)
        R.week()
        recs = gw.to_dict("records")
        for r in recs:
            neutral = r.get("location") == "Neutral"
            lh, la = R.predict(r["team"], r["team_a"], neutral)
            out = {"game_id": r["game_id"], "season": r["season"], "week": r["week"], "date": r["date"], "home": r["team"],
                   "away": r["team_a"], "playoff": r["season_type"] == "POST", "act_h": r["final"], "act_a": r["final_a"],
                   "pred_h": points(lh, ret) + lvl, "pred_a": points(la, ret) + lvl, "line_margin": r["spread_line"], "line_total": r["total_line"]}
            for k, c in enumerate(COMP): out[f"{c}_h"] = lh[k]; out[f"{c}_a"] = la[k]
            if hooks and "predict" in hooks: hooks["predict"](R, r, out)
            rows.append(out)
        res = np.mean([(o["act_h"] - o["pred_h"] + o["act_a"] - o["pred_a"]) / 2 for o in rows[-len(recs):]])
        lvl += prm.get("lvl_gain", 0.03) * min(len(recs) / prm.get("lvl_n", 16), 3) * res   # weight by games in the week
        for r in recs:
            neutral = r.get("location") == "Neutral"; s = 0 if neutral else 1
            R.update(r["team"], r["team_a"], s, np.array([r["y_" + c] for c in COMP]), np.array([r["n_" + c] for c in COMP]))
            R.update(r["team_a"], r["team"], -s, np.array([r["y_" + c + "_a"] for c in COMP]), np.array([r["n_" + c + "_a"] for c in COMP]))
    R.lvl, R.ret = lvl, ret
    return pd.DataFrame(rows), R
