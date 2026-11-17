"""Shot-state event tables from sportsdataverse NHL full play-by-play (2019-20 .. 2026-27).
Outputs (in OUT):
  shots.parquet   one row per unblocked+blocked shot attempt with state, xg, type, distance, rebound/rush flags, shooter, goalie
  segs.parquet    time (seconds) per game x team-perspective state (strength, score diff, period bucket, goalie pulled)
  games.parquet   final scores, OT/SO, date
Usage: python3 -I prep_events.py <dl dir> <out dir>"""
import sys, os, glob, numpy as np, pandas as pd
dl, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
COLS = ["event_type", "secondary_type", "event_team_abbr", "event_team_type", "period", "period_type", "game_seconds",
        "home_score", "away_score", "event_player_1_id", "event_goalie_id", "strength_state", "empty_net", "x_fixed", "y_fixed",
        "shot_distance", "shot_angle", "home_skaters", "away_skaters", "game_id", "season_type", "home_abbr", "away_abbr",
        "event_idx", "home_goalie_id", "away_goalie_id", "xg", "penalty_minutes", "penalty_severity"]
ATT = ["SHOT", "GOAL", "MISSED_SHOT", "BLOCKED_SHOT"]
S, G, T = [], [], []
seen = set()
for f in sorted(glob.glob(f"{dl}/pbp_*.parquet")):
    d = pd.read_parquet(f, columns=COLS)
    d = d[d.season_type.isin(["R", "P"]) & d.game_id.notna()]
    d = d[~d.game_id.isin(seen)]; seen |= set(d.game_id.unique())
    d = d.sort_values(["game_id", "period", "game_seconds", "event_idx"], kind="stable").reset_index(drop=True)
    d["season"] = d.game_id // 1000000
    # score BEFORE the event (pbp score columns include the goal itself on GOAL rows)
    isg = (d.event_type == "GOAL") & (d.period_type != "SHOOTOUT")
    hg = (isg & (d.event_team_type == "home")).astype(int); ag = (isg & (d.event_team_type == "away")).astype(int)
    d["hs0"] = hg.groupby(d.game_id).cumsum() - hg
    d["as0"] = ag.groupby(d.game_id).cumsum() - ag
    reg = d[d.period_type != "SHOOTOUT"].copy()
    reg["dt"] = (reg.groupby("game_id").game_seconds.shift(-1) - reg.game_seconds).clip(0, 300).fillna(0)
    hsk = reg.home_skaters.fillna(5).clip(3, 6); ask = reg.away_skaters.fillna(5).clip(3, 6)
    reg["h_pull"] = reg.home_goalie_id.isna() & (reg.period >= 3)
    reg["a_pull"] = reg.away_goalie_id.isna() & (reg.period >= 3)
    # skaters excluding the extra attacker -> manpower; pulled goalie flagged separately
    hm = hsk - reg.h_pull.astype(int); am = ask - reg.a_pull.astype(int)
    reg["hm"], reg["am"] = hm, am
    reg["pb"] = np.select([reg.period <= 2, reg.period == 3, reg.period > 3], ["p12", "p3", "ot"], "p12")
    reg["late"] = (reg.period == 3) & (reg.game_seconds >= 3000)  # last 10 min of regulation
    for side in ("home", "away"):
        o = "away" if side == "home" else "home"
        s = side[0]; oo = o[0]
        seg = pd.DataFrame({"game_id": reg.game_id, "season": reg.season, "side": side,
                            "team": reg[f"{side}_abbr"], "opp": reg[f"{o}_abbr"],
                            "man": reg[f"{s}m"].astype(int).astype(str) + "v" + reg[f"{oo}m"].astype(int).astype(str),
                            "sd": (reg[f"{s}s0"] - reg[f"{oo}s0"]).clip(-3, 3), "pb": reg.pb, "late": reg.late,
                            "pull": reg[f"{s}_pull"], "opull": reg[f"{oo}_pull"], "dt": reg.dt})
        T.append(seg.groupby(["game_id", "season", "side", "team", "opp", "man", "sd", "pb", "late", "pull", "opull"], as_index=False).dt.sum())
    a = reg[reg.event_type.isin(ATT)].copy()
    # previous event context for rebound / rush flags
    prev = reg.groupby("game_id")
    reg["p_t"] = prev.game_seconds.shift(1); reg["p_type"] = prev.event_type.shift(1)
    reg["p_team"] = prev.event_team_abbr.shift(1); reg["p_x"] = prev.x_fixed.shift(1); reg["p_y"] = prev.y_fixed.shift(1)
    a = reg[reg.event_type.isin(ATT)].copy()
    gap = a.game_seconds - a.p_t
    a["rebound"] = (gap <= 3) & a.p_type.isin(["SHOT", "MISSED_SHOT", "BLOCKED_SHOT"]) & (a.p_team == a.event_team_abbr)
    a["rush"] = (gap <= 4) & ~a.rebound & ((a.p_x.abs() < 25) | ((a.p_x * a.x_fixed) < 0)) & a.p_type.notna()
    is_h = a.event_team_type == "home"
    a["side"] = np.where(is_h, "home", "away")
    a["team"] = np.where(is_h, a.home_abbr, a.away_abbr); a["opp"] = np.where(is_h, a.away_abbr, a.home_abbr)
    a["man"] = np.where(is_h, a.hm.astype(int).astype(str) + "v" + a.am.astype(int).astype(str), a.am.astype(int).astype(str) + "v" + a.hm.astype(int).astype(str))
    a["sd"] = np.where(is_h, a.hs0 - a.as0, a.as0 - a.hs0).clip(-3, 3)
    a["pull"] = np.where(is_h, a.h_pull, a.a_pull); a["opull"] = np.where(is_h, a.a_pull, a.h_pull)
    a["goal"] = (a.event_type == "GOAL").astype(int); a["unbl"] = (a.event_type != "BLOCKED_SHOT").astype(int)
    a["sog"] = a.event_type.isin(["SHOT", "GOAL"]).astype(int)
    a = a.rename(columns={"event_player_1_id": "shooter", "event_goalie_id": "goalie", "secondary_type": "stype", "shot_distance": "dist"})
    S.append(a[["game_id", "season", "period", "game_seconds", "side", "team", "opp", "man", "sd", "pb", "late", "pull", "opull",
                "event_type", "goal", "unbl", "sog", "xg", "stype", "dist", "shot_angle", "rebound", "rush", "shooter", "goalie", "p_type", "p_team", "p_t", "p_x", "p_y", "x_fixed", "y_fixed"]])
    # games: final score from goals (shootout winner +1)
    so = d[(d.period_type == "SHOOTOUT") & (d.event_type == "GOAL")].groupby(["game_id", "event_team_type"]).size().unstack(fill_value=0)
    g = reg.groupby("game_id").agg(season=("season", "first"), st=("season_type", "first"), home=("home_abbr", "first"), away=("away_abbr", "first"),
                                   maxp=("period", "max"))
    g["hg"] = hg.groupby(d.game_id).sum().reindex(g.index); g["ag"] = ag.groupby(d.game_id).sum().reindex(g.index)
    g["so"] = g.index.isin(so.index)
    sh = so.reindex(g.index).fillna(0)
    g["h_so"] = sh.get("home", 0); g["a_so"] = sh.get("away", 0)
    G.append(g.reset_index())
    print(os.path.basename(f), len(g), len(a), flush=True)
pd.concat(S).to_parquet(f"{OUT}/shots.parquet"); pd.concat(T).to_parquet(f"{OUT}/segs.parquet")
games = pd.concat(G)
dates = []
for f in sorted(glob.glob(f"{dl}/pbp_*.parquet")):
    import pyarrow.parquet as pq
    if "game_date" in pq.ParquetFile(f).schema.names:
        dates.append(pd.read_parquet(f, columns=["game_id", "game_date"]).drop_duplicates("game_id"))
dates = pd.concat(dates).drop_duplicates("game_id")
games = games.merge(dates, on="game_id", how="left"); games.to_parquet(f"{OUT}/games.parquet")
print("done", len(games))
