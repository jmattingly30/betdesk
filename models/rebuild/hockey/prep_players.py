"""Player-game table from play-by-play on-ice columns: for every skater in every game, 5v5 time on ice and the
5v5 expected goals for and against while he was on the ice (empty-net shots excluded), plus all-strength TOI.
Usage: python3 prep_players.py <pbp download dir> [files...]   -> data/players_nhl/<file>.parquet"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/data/players_nhl"; os.makedirs(OUT, exist_ok=True)
dl = sys.argv[1]; files = sys.argv[2:] or sorted(glob.glob(f"{dl}/pbp_*.parquet"))
ON = [f"{s}_on_{i}_id" for s in ("home", "away") for i in range(1, 8)]
for f in files:
    f = f if "/" in f else f"{dl}/{f}"
    cols = ["game_id", "event_idx", "game_seconds", "period_type", "event_type", "strength_state", "empty_net", "xg",
            "event_team_abbr", "home_abbr", "away_abbr", "home_goalie_id", "away_goalie_id"] + ON
    d = pd.read_parquet(f, columns=cols)
    d = d[d.game_id.notna() & (d.period_type != "SHOOTOUT")].sort_values(["game_id", "event_idx"]).reset_index(drop=True)
    d["dt"] = (d.groupby("game_id").game_seconds.shift(-1) - d.game_seconds).clip(0, 300).fillna(0)
    ev5 = d.strength_state == "5v5"
    shot = d.xg.notna() & (d.empty_net != True) & ev5
    d["xh"] = np.where(shot & (d.event_team_abbr == d.home_abbr), d.xg, 0.0)
    d["xa"] = np.where(shot & (d.event_team_abbr == d.away_abbr), d.xg, 0.0)
    d["t5"] = np.where(ev5, d.dt, 0.0)
    parts = []
    for side, gcol, xf, xa, team in (("home", "home_goalie_id", "xh", "xa", "home_abbr"), ("away", "away_goalie_id", "xa", "xh", "away_abbr")):
        for i in range(1, 8):
            c = f"{side}_on_{i}_id"
            m = d[c].notna() & (d[c] != d[gcol])
            parts.append(pd.DataFrame({"game_id": d.game_id[m], "team": d[team][m], "player": d[c][m],
                                       "toi": d.dt[m], "toi5": d.t5[m], "xgf5": d[xf][m], "xga5": d[xa][m]}))
    p = pd.concat(parts).groupby(["game_id", "team", "player"], as_index=False).sum()
    p["game_id"] = p.game_id.astype("int64"); p["player"] = p.player.astype("int64")
    p.to_parquet(f"{OUT}/{os.path.basename(f)}"); print(os.path.basename(f), len(p), p.game_id.nunique(), round(p.toi.mean() / 60, 1), flush=True)
