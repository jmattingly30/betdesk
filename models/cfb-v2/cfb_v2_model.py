"""
CFB v2 play-level (PPA) model, standalone and portable.

One-file port of the validated v2 pipeline (pl_consolidate -> pl_walkforward -> pl_priors ->
pl_build_features -> pl_engine + bt_engine probability layer), plus the betting rules found in
the edge-finding work. See CFB_V2_MODEL_GUIDE.md for the step-by-step explanation.

Usage (weekly):
    from cfb_v2_model import CFBD, run_week
    cfbd = CFBD()                      # reads CFBD_API_KEY env var (or pass api_key=...)
    slate = run_week(cfbd, season=2026, week=7)
    slate[slate.spread_bet | slate.total_bet]

Requires: numpy, pandas, scipy, scikit-learn, requests
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm
from sklearn.linear_model import RidgeCV

# =============================================================================================
# 1. CONSTANTS AND FITTED SETTINGS (fit on 2021-2023 only; validated on 2024-2026wk3 holdout)
# =============================================================================================
SETTINGS = {
    "shrink": 1.0,                       # fitted value 1.0 means no shrink toward league average
    "hfa": 3.6295,                       # home-field advantage, points (split +/- hfa/2)
    "points_baseline": 18.9344,          # per-team points offset; PPA is value ABOVE expectation
    # weight on preseason prior by # FBS-vs-FBS games played (index 0..7, 7 = 7+)
    "prior_table": [0.9830, 0.9560, 0.4690, 0.4690, 0.4690, 0.2664, 0.1240, 0.1240],
    # margin sigma by |projected margin| bucket: [0,7) [7,14) [14,21) [21,28) [28,40) [40,inf)
    "sigma_table": [16.5047, 17.4908, 19.4265, 18.6601, 17.0150, 17.5781],
    "total_sigma": 16.7759,
    "share_1h": 0.51696,
    "share_1q": 0.21969,
    "sigma_1h_margin": 12.7098, "sigma_1h_total": 11.2668,
    "sigma_1q_margin": 8.2147, "sigma_1q_total": 7.3047,
}
SIGMA_BOUNDS = np.array([0, 7, 14, 21, 28, 40])

# FCS / non-FBS opponents get one fixed rating (empirical 2021-2023 means)
FCS_OFF_PPA, FCS_DEF_PPA, FCS_PLAYS = -0.105, 0.382, 55.7

RIDGE_ALPHAS = np.logspace(-3, 4, 25)
RECENCY_DECAY = 0.95                    # weight = 0.95 ** weeks_ago

# Key-number multipliers for exact final margins 0..100, fit on 2021-2023
# (empirical frequency / reference-normal probability).
KEY_MULT_FITTED = np.array([
    0.0, 0.7643, 0.9205, 2.8225, 0.934, 0.5395, 0.9227, 2.4531, 0.8012, 0.4811, 1.2799, 0.5933,
    0.4542, 0.7624, 1.4787, 0.6584, 0.2675, 1.438, 0.8872, 0.4723, 0.7426, 1.5792, 0.548, 0.6199,
    1.4216, 0.7657, 0.5429, 1.0261, 1.912, 0.6627, 0.5499, 1.6366, 1.0145, 0.7316, 1.2339, 1.437,
    0.5219, 0.855, 1.8098, 0.8219, 0.4521, 0.7478, 2.2964, 0.1018, 0.9045, 2.1403, 0.8432, 0.4717,
    0.7054, 2.9743, 0.8939, 0.2525, 2.8592, 1.623, 0.0, 1.6854, 3.3728, 1.1046, 1.2692, 2.1928,
    0.8439, 0.0, 2.2665, 1.318, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 13.6276]
    + [0.0] * 27)
KEY_MULT = KEY_MULT_FITTED
# TESTED AND REJECTED (2026-10-10): the tail past |margin| 35 looks noisy (zeros at 54, 61, 64+;
# 13.6x at 73), but flattening it to 1.0 made Brier scores slightly WORSE on both train and
# holdout (cover 0.2705 -> 0.2719 holdout; win prob 0.1692 -> 0.1693). Kept only for reference.
# The betting signals use point edges and never touch this table either way.
KEY_MULT_TAIL_SMOOTHED = KEY_MULT_FITTED.copy()
KEY_MULT_TAIL_SMOOTHED[36:] = 1.0
MARGIN_GRID = np.arange(-100, 101)

# Betting rules (from the edge analysis). Pick against the OPENING line and bet early.
# Spread bands: (min |open spread|, max |open spread|, min |model edge| in points)
SPREAD_BANDS = [(0.0, 6.0, 7.5), (6.5, 13.5, 9.0), (14.0, 20.5, 9.5), (21.0, 27.5, 5.0)]
# 28+ band intentionally excluded: lost in train and holdout, negative CLV.
TOTAL_EDGE_MIN = 10.5
EARLY_SEASON_LAST_WEEK = 4   # see add_signals(): early-season bets are much weaker
PROVIDER_PREF = ["consensus", "DraftKings", "Bovada", "ESPN Bet", "William Hill (New Jersey)", "teamrankings"]


# =============================================================================================
# 2. CFBD DATA ACCESS (cached, so each endpoint+params is only ever downloaded once)
# =============================================================================================
class CFBD:
    BASE = "https://api.collegefootballdata.com"

    def __init__(self, api_key=None, cache_dir="cfbd_cache"):
        key = api_key or os.environ.get("CFBD_API_KEY")
        if not key:
            raise ValueError("Set CFBD_API_KEY env var or pass api_key=")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.calls = 0

    def get(self, endpoint, params, use_cache=True):
        h = hashlib.sha256((endpoint + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:24]
        path = os.path.join(self.cache_dir, f"{endpoint.strip('/').replace('/', '_')}_{h}.json")
        if use_cache and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        for attempt in range(3):
            r = self.session.get(self.BASE + endpoint, params=params, timeout=60)
            self.calls += 1
            if r.status_code == 200:
                data = r.json()
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"CFBD {endpoint} {params} -> HTTP {r.status_code}: {r.text[:300]}")


def load_season(cfbd, season, refresh_current=False):
    """Whole-season pulls (omitting `week` is ~15x cheaper than per-week loops).
    refresh_current=True re-downloads (use for the in-progress season so new weeks appear)."""
    uc = not refresh_current
    games = (cfbd.get("/games", {"year": season, "seasonType": "regular"}, uc)
             + cfbd.get("/games", {"year": season, "seasonType": "postseason"}, uc))
    adv = cfbd.get("/stats/game/advanced",
                   {"year": season, "seasonType": "both", "excludeGarbageTime": "true"}, uc)
    lines = (cfbd.get("/lines", {"year": season, "seasonType": "regular"}, uc)
             + cfbd.get("/lines", {"year": season, "seasonType": "postseason"}, uc))
    teams = cfbd.get("/teams", {"year": season}, True)
    returning = cfbd.get("/player/returning", {"year": season}, True)
    class_map = {t["school"]: t.get("classification") for t in teams}
    return {"season": season, "games": games, "adv": adv, "lines": lines,
            "class_map": class_map, "returning": returning}


# =============================================================================================
# 3. TEAM-GAME TABLE: one row per team per game (offense = that team)
# =============================================================================================
def games_frame(raw):
    rows = []
    for g in raw["games"]:
        rows.append({"gameId": g["id"], "season": g["season"], "week": g["week"],
                     "seasonType": g["seasonType"], "neutral": bool(g.get("neutralSite", False)),
                     "completed": bool(g.get("completed", False)),
                     "home": g.get("homeTeam"), "away": g.get("awayTeam"),
                     "homePoints": g.get("homePoints"), "awayPoints": g.get("awayPoints"),
                     "startDate": g.get("startDate")})
    return pd.DataFrame(rows).drop_duplicates("gameId")


def team_game_table(raw):
    meta = games_frame(raw).set_index("gameId")
    cm = raw["class_map"]
    rows = []
    for r in raw["adv"]:
        gid = r.get("gameId")
        if gid not in meta.index:
            continue
        m = meta.loc[gid]
        team, opp, off = r.get("team"), r.get("opponent"), (r.get("offense") or {})
        if team == m["home"]:
            is_home = True
        elif team == m["away"]:
            is_home = False
        else:
            continue
        rows.append({"gameId": gid, "season": m["season"], "week": m["week"],
                     "seasonType": m["seasonType"], "offense": team, "defense": opp,
                     "home_ind": 0 if m["neutral"] else (1 if is_home else -1),
                     "off_ppa": off.get("ppa"), "off_plays": off.get("plays")})
    df = pd.DataFrame(rows).dropna(subset=["off_ppa", "off_plays"])
    df["fbs_vs_fbs"] = (df["offense"].map(cm) == "fbs") & (df["defense"].map(cm) == "fbs")
    return df


# =============================================================================================
# 4. RATINGS: weekly walk-forward ridge opponent adjustment
# =============================================================================================
def _fit_ridge(data):
    X = pd.concat([pd.get_dummies(data["offense"], prefix="off", dtype=float),
                   pd.get_dummies(data["defense"], prefix="def", dtype=float)], axis=1)
    X["home_ind"] = data["home_ind"].astype(float).values
    y = data["off_ppa"].astype(float).values
    w = data["weight"].astype(float).values
    model = RidgeCV(alphas=RIDGE_ALPHAS, fit_intercept=True).fit(X.values, y, sample_weight=w)
    return pd.Series(model.coef_, index=X.columns), float(np.average(y, weights=w)), float(model.alpha_)


def _snapshot(reg, plays_rows, teams_universe):
    """reg: FBS-vs-FBS rows used in the regression (with weights). plays_rows: all rows (incl.
    FCS games) used for plays/game."""
    snap = {"teams": {}, "league_avg_ppa": None, "league_avg_plays": None, "n_games": len(reg), "alpha": None}
    if len(reg) == 0:
        return snap
    coefs, avg, alpha = _fit_ridge(reg)
    snap.update(league_avg_ppa=avg, alpha=alpha)
    plays_vals = []
    for team in teams_universe:
        n_c = reg.loc[(reg["offense"] == team) | (reg["defense"] == team), "gameId"].nunique()
        orows = plays_rows[plays_rows["offense"] == team]
        L = float(orows["off_plays"].mean()) if len(orows) else None
        if L is not None:
            plays_vals.append(L)
        snap["teams"][team] = {
            "C": int(n_c),                                                     # FBS games played
            "H": avg + coefs[f"off_{team}"] if f"off_{team}" in coefs.index else None,  # off PPA
            "J": avg + coefs[f"def_{team}"] if f"def_{team}" in coefs.index else None,  # def PPA allowed
            "L": L}                                                            # plays/game
    snap["league_avg_plays"] = float(np.mean(plays_vals)) if plays_vals else None
    return snap


def rate_teams(tg, before_week):
    """Ratings using ONLY regular-season games with week < before_week (no leakage)."""
    teams_universe = sorted(set(tg["offense"]) | set(tg["defense"]))
    regular = tg[tg["seasonType"] == "regular"]
    reg = regular[regular["fbs_vs_fbs"] & (regular["week"] < before_week)].copy()
    reg["weight"] = RECENCY_DECAY ** (before_week - reg["week"])
    return _snapshot(reg, regular[regular["week"] < before_week], teams_universe)


def rate_teams_end_of_season(tg, fix_postseason_weight=False):
    """Full-season ratings (incl. bowls), used to build NEXT season's priors.
    Note: CFBD tags bowl games week=1, so the original v2 weighted bowls like week-1 games
    (low weight). fix_postseason_weight=True treats them as the most recent week instead. That
    is more sensible, but it is UNTESTED and changes priors slightly. Default keeps the
    validated behavior."""
    teams_universe = sorted(set(tg["offense"]) | set(tg["defense"]))
    pseudo = int(tg.loc[tg["seasonType"] == "regular", "week"].max()) + 1
    reg = tg[tg["fbs_vs_fbs"]].copy()
    wk = reg["week"].clip(upper=pseudo)
    if fix_postseason_weight:
        wk = np.where(reg["seasonType"] == "postseason", pseudo, wk)
    reg["weight"] = RECENCY_DECAY ** np.clip(pseudo - wk, 0, None)
    return _snapshot(reg, tg, teams_universe)


# =============================================================================================
# 5. PRESEASON PRIORS: last season's ratings regressed toward average by returning production
# =============================================================================================
def build_priors(prev_eos, returning):
    if not prev_eos or prev_eos.get("league_avg_ppa") is None:
        return {}
    avg = prev_eos["league_avg_ppa"]
    rp = {r["team"]: r.get("percentPPA") for r in (returning or [])}
    priors = {}
    for team, v in prev_eos["teams"].items():
        if v["H"] is None:
            continue
        pct = rp.get(team)
        f = 0.55 + 0.35 * pct if pct is not None else 0.6
        # CFBD has no defensive returning production; offensive percentPPA is used for both.
        priors[team] = {"D": avg + (v["H"] - avg) * f, "F": avg + (v["J"] - avg) * f}
    return priors


# =============================================================================================
# 6. PROJECTION: ratings -> points -> margin/total -> probabilities
# =============================================================================================
def _blend(prior, season, default, M):
    prior, season = np.asarray(prior, float), np.asarray(season, float)
    pm, sm = np.isnan(prior), np.isnan(season)
    return np.where(pm & sm, default, np.where(sm, prior, np.where(pm, season, M * prior + (1 - M) * season)))


def _side(C, D, H, F, J, is_fbs, s, avg):
    M = np.array(s["prior_table"])[np.clip(np.nan_to_num(C, nan=0).astype(int), 0, 7)]
    off = s["shrink"] * _blend(D, H, FCS_OFF_PPA, M) + (1 - s["shrink"]) * avg
    dfn = s["shrink"] * _blend(F, J, FCS_DEF_PPA, M) + (1 - s["shrink"]) * avg
    return np.where(is_fbs, off, FCS_OFF_PPA), np.where(is_fbs, dfn, FCS_DEF_PPA)


def _sigma(abs_margin, table):
    return np.array(table)[np.clip(np.searchsorted(SIGMA_BOUNDS, abs_margin, side="right") - 1, 0, len(table) - 1)]


def _key_probs(mu, sigma, key_mult, offset=None):
    """Discrete margin distribution: normal mass at each integer margin x key-number weight,
    renormalized. offset=None -> P(home wins). Else (P(home covers), P(push)) for spread=offset."""
    mu, sigma = np.asarray(mu, float)[:, None], np.asarray(sigma, float)[:, None]
    g = MARGIN_GRID[None, :]
    wt = (norm.cdf(g + 0.5, mu, sigma) - norm.cdf(g - 0.5, mu, sigma)) * key_mult[np.abs(MARGIN_GRID)][None, :]
    tot = wt.sum(axis=1)
    if offset is None:
        return (wt * (g > 0)).sum(axis=1) / tot
    o = np.asarray(offset, float)[:, None]
    return (wt * ((g + o) > 0)).sum(axis=1) / tot, (wt * ((g + o) == 0)).sum(axis=1) / tot


def _over_prob(line, mu, sigma):
    line = np.asarray(line, float)
    return 1 - norm.cdf(np.floor(line) + 0.5, mu, sigma)


def project(df, settings=SETTINGS, key_mult=KEY_MULT):
    """df needs, per side (home_/away_): C, D, H, F, J, L ratings; plus homeClass, awayClass,
    neutral, league_avg_ppa, league_avg_plays, and optionally spreadOpen / overUnderOpen.
    Returns df plus proj_home, proj_away, proj_margin (home - away), proj_total, win_prob_home,
    cover/over probabilities, and 1H/1Q projections."""
    s, out = settings, df.copy()
    hf, af = (df["homeClass"] == "fbs").values, (df["awayClass"] == "fbs").values
    avg = df["league_avg_ppa"].values.astype(float)
    avg_plays = df["league_avg_plays"].values.astype(float)
    col = lambda side, k: df[f"{side}_{k}"].values.astype(float)

    off_h, def_h = _side(col("home", "C"), col("home", "D"), col("home", "H"), col("home", "F"), col("home", "J"), hf, s, avg)
    off_a, def_a = _side(col("away", "C"), col("away", "D"), col("away", "H"), col("away", "F"), col("away", "J"), af, s, avg)
    plays_h = np.where(hf, np.where(np.isnan(col("home", "L")), avg_plays, col("home", "L")), FCS_PLAYS)
    plays_a = np.where(af, np.where(np.isnan(col("away", "L")), avg_plays, col("away", "L")), FCS_PLAYS)
    plays = (plays_h + plays_a) / 2

    # expected PPA/play for each offense vs that defense (additive matchup, re-centred)
    ppa_home = off_h + def_a - avg
    ppa_away = off_a + def_h - avg
    hfa_half = np.where(df["neutral"].astype(bool).values, 0.0, s["hfa"] / 2)
    home_pts = np.maximum(0, plays * ppa_home + s["points_baseline"] + hfa_half)
    away_pts = np.maximum(0, plays * ppa_away + s["points_baseline"] - hfa_half)
    margin, total = home_pts - away_pts, home_pts + away_pts
    sig = _sigma(np.abs(margin), s["sigma_table"])

    out["proj_home"], out["proj_away"] = home_pts, away_pts
    out["proj_margin"], out["proj_total"] = margin, total
    out["margin_sigma"] = sig
    out["win_prob_home"] = _key_probs(margin, sig, key_mult)
    for line_col, tag in [("spreadOpen", "open"), ("spread", "close")]:
        if line_col in df:
            ln = df[line_col].values.astype(float)
            have = ~np.isnan(ln)
            pc, pp = np.full(len(df), np.nan), np.full(len(df), np.nan)
            if have.any():
                pc[have], pp[have] = _key_probs(margin[have], sig[have], key_mult, ln[have])
            out[f"p_home_cover_{tag}"], out[f"p_push_{tag}"] = pc, pp
    for line_col, tag in [("overUnderOpen", "open"), ("overUnder", "close")]:
        if line_col in df:
            out[f"p_over_{tag}"] = _over_prob(df[line_col].values.astype(float), total, s["total_sigma"])
    out["proj_1h_margin"], out["proj_1h_total"] = margin * s["share_1h"], total * s["share_1h"]
    out["proj_1q_margin"], out["proj_1q_total"] = margin * s["share_1q"], total * s["share_1q"]
    return out


# =============================================================================================
# 7. BETTING SIGNALS (spread bands + totals), always against the OPENING line
# =============================================================================================
def spread_signal(proj_margin, spread_open):
    """spread_open is home-perspective (home favored = negative). Returns (bet?, side, edge)."""
    if spread_open is None or np.isnan(spread_open):
        return False, None, np.nan
    edge = proj_margin + spread_open          # >0: model likes home vs the number
    a = abs(spread_open)
    for lo, hi, thr in SPREAD_BANDS:
        if lo <= a <= hi:
            return abs(edge) >= thr, ("home" if edge > 0 else "away"), edge
    return False, None, edge                  # 28+ band or between-band values: no bet


def total_signal(proj_total, total_open):
    if total_open is None or np.isnan(total_open):
        return False, None, np.nan
    edge = proj_total - total_open
    return abs(edge) >= TOTAL_EDGE_MIN, ("over" if edge > 0 else "under"), edge


def add_signals(proj):
    line = lambda c: pd.to_numeric(proj[c], errors="coerce") if c in proj else pd.Series(np.nan, index=proj.index)
    for name, fn, col_proj, col_line in [("spread", spread_signal, "proj_margin", "spreadOpen"),
                                         ("total", total_signal, "proj_total", "overUnderOpen")]:
        res = [fn(x, o) for x, o in zip(proj[col_proj], line(col_line))]
        proj[f"{name}_bet"] = [r[0] for r in res]
        proj[f"{name}_side"] = [r[1] for r in res]
        proj[f"{name}_edge"] = [r[2] for r in res]
        proj[f"{name}_bet"] = proj[f"{name}_bet"].astype(bool)
    # Weeks 1-4 produced ~60% of spread bets but ran near breakeven (train 51.6%, holdout 53.3%);
    # weeks 5+ carried the edge (train 61.8%, holdout 56.9%). Flag them so stakes can differ.
    if "week" in proj:
        proj["early_season"] = pd.to_numeric(proj["week"], errors="coerce") <= EARLY_SEASON_LAST_WEEK
    return proj


# =============================================================================================
# 8. GLUE: lines, slate features, and the weekly runner
# =============================================================================================
def lines_frame(raw_lines):
    rows = []
    for e in raw_lines:
        lines = e.get("lines") or []
        byp = {ln["provider"]: ln for ln in lines}
        chosen = next((byp[p] for p in PROVIDER_PREF if p in byp and byp[p].get("spread") is not None), None)
        chosen = chosen or next((ln for ln in lines if ln.get("spread") is not None), None)
        if chosen is None:
            continue
        pick = lambda key: chosen.get(key) if chosen.get(key) is not None else next(
            (ln[key] for ln in lines if ln.get(key) is not None), None)
        rows.append({"gameId": e["id"], "spreadOpen": pick("spreadOpen"), "spread": chosen.get("spread"),
                     "overUnderOpen": pick("overUnderOpen"), "overUnder": pick("overUnder"),
                     "provider": chosen["provider"]})
    return pd.DataFrame(rows, columns=["gameId", "spreadOpen", "spread", "overUnderOpen", "overUnder", "provider"]
                        ).drop_duplicates("gameId")


def slate_features(games, snap, priors, class_map, fallback_avgs):
    """games: rows of games_frame. snap: rate_teams(...) output for the slate week."""
    avg_ppa, avg_plays = snap.get("league_avg_ppa"), snap.get("league_avg_plays")
    if avg_ppa is None or snap.get("n_games", 0) == 0:          # week 1: use last season's
        avg_ppa, avg_plays = fallback_avgs
    rows = []
    for _, g in games.iterrows():
        hc, ac = class_map.get(g["home"]), class_map.get(g["away"])
        if hc != "fbs" and ac != "fbs":
            continue                                            # FCS-vs-FCS not modeled
        row = {**g.to_dict(), "homeClass": hc, "awayClass": ac,
               "league_avg_ppa": avg_ppa, "league_avg_plays": avg_plays}
        for side, team in [("home", g["home"]), ("away", g["away"])]:
            t, p = snap["teams"].get(team, {}), priors.get(team, {})
            row.update({f"{side}_C": t.get("C", 0), f"{side}_H": t.get("H"), f"{side}_J": t.get("J"),
                        f"{side}_L": t.get("L"), f"{side}_D": p.get("D"), f"{side}_F": p.get("F")})
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in [f"{s}_{k}" for s in ("home", "away") for k in "CDFHJL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def run_week(cfbd, season, week, refresh=True):
    """Project every FBS-involved regular-season game in (season, week) and flag bets.
    Uses only games before `week` for ratings. API cost: ~12 calls on the first run; after that
    ~5 per run (refresh=True re-pulls the current season's games/stats/lines so new weeks and
    opening lines appear; the prior season, teams, and returning production stay cached)."""
    raw = load_season(cfbd, season, refresh_current=refresh)
    prev = load_season(cfbd, season - 1)
    tg, tg_prev = team_game_table(raw), team_game_table(prev)

    prev_eos = rate_teams_end_of_season(tg_prev)
    priors = build_priors(prev_eos, raw["returning"])
    snap = rate_teams(tg, before_week=week)

    games = games_frame(raw)
    games = games[(games["week"] == week) & (games["seasonType"] == "regular")]
    games = games.merge(lines_frame(raw["lines"]), on="gameId", how="left")
    feats = slate_features(games, snap, priors, raw["class_map"],
                           (prev_eos["league_avg_ppa"] or 0.12, prev_eos["league_avg_plays"] or 62.0))
    return add_signals(project(feats))
