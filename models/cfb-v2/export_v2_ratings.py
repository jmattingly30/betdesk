"""Weekly export of J's CFB v2 model for the Bet Desk app (Games tab > CFB > v2).

Run on your own computer (CollegeFootballData is not reachable from the cloud), in the folder that
holds cfb_v2_model.py:

    export CFBD_API_KEY=...            (Windows: set CFBD_API_KEY=...)
    python export_v2_ratings.py 2026 8 (season, the week you are about to bet)

Writes cfb_v2.json: every team's v2 offense/defense PPA for that week (prior already blended in, exactly
as cfb_v2_model.project does), plays per game, the v2 settings and key-number table, and v2's own slate
for the week with its bet flags. Put the file in the CFB v2 Drive folder and ask Claude to load it.
Uses only cfb_v2_model's own functions, so the app matches run_week.
"""
import datetime
import json
import sys

import numpy as np

import cfb_v2_model as v2


def _f(x):
    return np.nan if x is None else float(x)


def _r(x, d=5):
    return None if x is None or not np.isfinite(x) else round(float(x), d)


def main(season, week, out="cfb_v2.json"):
    cfbd = v2.CFBD()
    raw = v2.load_season(cfbd, season, refresh_current=True)
    prev = v2.load_season(cfbd, season - 1)
    tg, tg_prev = v2.team_game_table(raw), v2.team_game_table(prev)
    prev_eos = v2.rate_teams_end_of_season(tg_prev)
    priors = v2.build_priors(prev_eos, raw["returning"])
    snap = v2.rate_teams(tg, before_week=week)
    avg_ppa, avg_plays = snap.get("league_avg_ppa"), snap.get("league_avg_plays")
    if avg_ppa is None or snap.get("n_games", 0) == 0:          # week 1: last season's, as run_week does
        avg_ppa, avg_plays = prev_eos["league_avg_ppa"] or 0.12, prev_eos["league_avg_plays"] or 62.0
    cm = raw["class_map"]

    teams, other = [], []
    for name in sorted(set(cm) | set(snap["teams"]) | set(priors)):
        if cm.get(name) != "fbs":
            other.append(name)
            continue
        t, p = snap["teams"].get(name, {}), priors.get(name, {})
        c = int(t.get("C", 0) or 0)
        off, dfn = v2._side(np.array([c], float), np.array([_f(p.get("D"))]), np.array([_f(t.get("H"))]),
                            np.array([_f(p.get("F"))]), np.array([_f(t.get("J"))]), np.array([True]),
                            v2.SETTINGS, np.array([float(avg_ppa)]))
        L = t.get("L")
        teams.append([name, _r(off[0]), _r(dfn[0]), _r(L, 2) if L is not None else None, c, 1])

    slate = v2.run_week(cfbd, season, week, refresh=False)       # same data, already cached above
    keep = ["home", "away", "neutral", "spreadOpen", "overUnderOpen", "proj_margin", "proj_total",
            "spread_bet", "spread_side", "spread_edge", "total_bet", "total_side", "total_edge"]
    rows = []
    for _, g in slate.iterrows():
        r = {}
        for k in keep:
            v = g.get(k)
            if isinstance(v, (bool, np.bool_)):
                r[k] = bool(v)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                r[k] = _r(float(v), 2)
            else:
                r[k] = None if v is None or (isinstance(v, float) and np.isnan(v)) else v
        rows.append(r)

    doc = {"model": "CFB v2", "season": season, "week": week,
           "made": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
           "league_avg_ppa": round(float(avg_ppa), 5), "league_avg_plays": round(float(avg_plays), 2),
           "settings": v2.SETTINGS, "key_mult": [round(float(x), 4) for x in v2.KEY_MULT],
           "fcs": [v2.FCS_OFF_PPA, v2.FCS_DEF_PPA, v2.FCS_PLAYS],
           "spread_bands": [list(b) for b in v2.SPREAD_BANDS], "total_edge_min": v2.TOTAL_EDGE_MIN,
           "early_last_week": v2.EARLY_SEASON_LAST_WEEK,
           "teams": teams, "other": other, "slate": rows}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    bets = sum(bool(r.get("spread_bet")) + bool(r.get("total_bet")) for r in rows)
    print(f"wrote {out}: {len(teams)} FBS teams, {len(rows)} games in week {week}, {bets} v2 bets flagged")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python export_v2_ratings.py SEASON WEEK")
    main(int(sys.argv[1]), int(sys.argv[2]))
