"""NFL drive table from nflverse play-by-play: one row per drive with the offense, start field position and clock,
what the drive did (plays, EPA, success, explosives, sacks, turnovers), how it ended, and the points each side
scored during it (offense: TD + try or FG, defense: return TDs and safeties). Also a team-game table.
Usage: python3 prep_drives.py <pbp dir>   -> data/drives_nfl.parquet, data/teamgames_nfl.parquet"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); dl = sys.argv[1]
C = ["game_id", "season", "season_type", "week", "game_date", "home_team", "away_team", "posteam", "defteam", "fixed_drive",
     "fixed_drive_result", "play_type", "yardline_100", "game_seconds_remaining", "game_half", "qtr", "epa", "success",
     "pass", "rush", "yards_gained", "sack", "interception", "fumble_lost", "total_home_score", "total_away_score",
     "score_differential", "qb_kneel", "qb_spike", "drive_start_transition", "penalty", "posteam_type"]
out = []
for f in sorted(glob.glob(f"{dl}/pbp_*.parquet")):
    d = pd.read_parquet(f, columns=C)
    d = d[d.fixed_drive.notna()].copy()
    d["dh"] = d.groupby("game_id").total_home_score.diff().fillna(d.total_home_score).clip(lower=0)
    d["da"] = d.groupby("game_id").total_away_score.diff().fillna(d.total_away_score).clip(lower=0)
    # the drive's offense is the team on offense for most scrimmage plays (kickoff rows carry the receiving team)
    scrim = d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1)
    d["scrim"] = scrim.astype(int)
    d["expl"] = ((d["pass"] == 1) & (d.yards_gained >= 20) | (d["rush"] == 1) & (d.yards_gained >= 10)).astype(int) * d.scrim
    d["to"] = (d.interception.fillna(0) + d.fumble_lost.fillna(0)).clip(upper=1)
    key = ["game_id", "fixed_drive"]
    g = d.groupby(key, sort=False)
    first_scrim = d[scrim].groupby(key).agg(start_y100=("yardline_100", "first"), start_sec=("game_seconds_remaining", "first"),
                                             start_diff=("score_differential", "first"))
    agg = g.agg(season=("season", "first"), season_type=("season_type", "first"), week=("week", "first"), date=("game_date", "first"),
                home=("home_team", "first"), away=("away_team", "first"), off=("posteam", lambda s: s.mode().iat[0] if s.notna().any() else None),
                result=("fixed_drive_result", "first"), half=("game_half", "first"), qtr=("qtr", "first"),
                sec0=("game_seconds_remaining", "first"), sec1=("game_seconds_remaining", "last"),
                dh=("dh", "sum"), da=("da", "sum"), trans=("drive_start_transition", "first"))
    s = d[scrim].groupby(key).agg(plays=("scrim", "sum"), epa=("epa", "sum"), succ=("success", "sum"), expl=("expl", "sum"),
                                  sacks=("sack", "sum"), passes=("pass", "sum"), turnovers=("to", "sum"))
    a = agg.join(first_scrim).join(s).reset_index()
    a = a[a.off.notna()]
    a["deff"] = np.where(a.off == a.home, a.away, a.home); a["is_home"] = (a.off == a.home).astype(int)
    a["pts_off"] = np.where(a.is_home == 1, a.dh, a.da); a["pts_def"] = np.where(a.is_home == 1, a.da, a.dh)
    a["dur"] = (a.sec0 - a.sec1).clip(lower=0)
    out.append(a.drop(columns=["dh", "da"])); print(os.path.basename(f), len(a), flush=True)
D = pd.concat(out, ignore_index=True)
D["date"] = pd.to_datetime(D.date)
for c in ("plays", "epa", "succ", "expl", "sacks", "passes", "turnovers"): D[c] = D[c].fillna(0)
D.to_parquet(f"{HERE}/data/drives_nfl.parquet")
# team-game table (offense view)
T = D.groupby(["game_id", "off"]).agg(season=("season", "first"), season_type=("season_type", "first"), week=("week", "first"),
        date=("date", "first"), deff=("deff", "first"), is_home=("is_home", "first"), drives=("result", "size"),
        pts_off=("pts_off", "sum"), plays=("plays", "sum"), epa=("epa", "sum"), succ=("succ", "sum"), expl=("expl", "sum"),
        turnovers=("turnovers", "sum"), start_y100=("start_y100", "mean"), dur=("dur", "sum")).reset_index().rename(columns={"off": "team", "deff": "opp"})
pd_ = D.groupby(["game_id", "deff"]).pts_def.sum().rename("pts_ret").reset_index().rename(columns={"deff": "team"})
T = T.merge(pd_, on=["game_id", "team"], how="left").fillna({"pts_ret": 0}); T["pts"] = T.pts_off + T.pts_ret
gm = pd.read_csv(f"{dl}/games.csv")[["game_id", "home_score", "away_score", "home_team", "away_team", "location", "spread_line", "total_line", "roof", "temp", "wind", "home_qb_id", "away_qb_id"]]
T = T.merge(gm, on="game_id", how="left")
T["final"] = np.where(T.is_home == 1, T.home_score, T.away_score)
print("points match final:", (T.pts == T.final).mean().round(4), "rows", len(T))
T.to_parquet(f"{HERE}/data/teamgames_nfl.parquet")
