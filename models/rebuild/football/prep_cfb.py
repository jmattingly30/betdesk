"""CFB team-game table (same columns as the NFL one) from ESPN play-by-play (sportsdataverse espn_cfb_pbp) plus the
project's games table (dates, neutral site, closing lines). Usage: python3 prep_cfb.py <espn pbp dir>"""
import sys, os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); dl = sys.argv[1]
C = ["game_id", "season", "pos_team_id", "def_pos_team_id", "scrimmage_play", "EPA", "EPA_success", "pass", "rush", "statYardage",
     "turnover_vec", "drive.id", "start.yardsToEndzone", "wp_before", "homeTeamId", "awayTeamId", "homeFinalScore", "awayFinalScore"]
out = []
for f in sorted(glob.glob(f"{dl}/espn_*.parquet")):
    d = pd.read_parquet(f, columns=C)
    d = d[(d.scrimmage_play == True) & d.pos_team_id.notna()].copy()
    d["expl"] = (((d["pass"] == True) & (d.statYardage >= 20)) | ((d["rush"] == True) & (d.statYardage >= 10))).astype(int)
    k = ["game_id", "pos_team_id"]
    if os.environ.get("NG"):   # efficiency from competitive plays only (win prob 10-90%); drives/points from all plays
        d["ng"] = d.wp_before.between(0.1, 0.9)
        for c in ("EPA", "EPA_success", "expl", "turnover_vec"): d[c] = d[c].astype(float).where(d.ng)
    dr = d.groupby(k + ["drive.id"])["start.yardsToEndzone"].first().reset_index()
    g = d.groupby(k).agg(season=("season", "first"), opp=("def_pos_team_id", "first"), plays=("EPA", "count"), epa=("EPA", "sum"),
                         succ=("EPA_success", "sum"), expl=("expl", "sum"), turnovers=("turnover_vec", "sum"),
                         home_id=("homeTeamId", "first"), hf=("homeFinalScore", "max"), af=("awayFinalScore", "max")).reset_index()
    g = g.merge(dr.groupby(k).agg(drives=("drive.id", "size"), start_y100=("start.yardsToEndzone", "mean")).reset_index(), on=k)
    out.append(g); print(os.path.basename(f), len(g), flush=True)
T = pd.concat(out, ignore_index=True).rename(columns={"pos_team_id": "team"})
T["is_home"] = (T.team == T.home_id).astype(int)
G = pd.read_parquet(os.path.join(HERE, "../../team-strength/data/cfb_games.parquet"))
G = G[["game_id", "week", "season_type", "start_date", "neutral_site", "line_close", "total_close", "home_points", "away_points", "home_division", "away_division"]]
T = T.merge(G, on="game_id", how="inner")
T["final"] = np.where(T.is_home == 1, T.home_points, T.away_points).astype(float)
T["final"] = T.final.fillna(pd.Series(np.where(T.is_home == 1, T.hf, T.af), index=T.index))
T["pts_off"] = T.final; T["pts_ret"] = 0.0
T["date"] = pd.to_datetime(T.start_date, utc=True).dt.tz_localize(None)
T["season_type"] = np.where(T.season_type.eq("postseason"), "POST", "REG")
T["week"] = np.where(T.season_type == "POST", 20, T.week)
T["location"] = np.where(T.neutral_site == True, "Neutral", "Home")
T["spread_line"] = T.line_close; T["total_line"] = T.total_close
T = T[T.groupby("game_id").team.transform("size") == 2]
T["team"] = T.team.astype(int); T["opp"] = T.opp.astype(int)
T.drop(columns=["hf", "af", "home_id", "start_date"]).to_parquet(f"{HERE}/data/teamgames_cfb{os.environ.get('NG', '') and '_ng'}.parquet")
print(len(T), T.season.min(), T.season.max())
h = T[T.is_home == 1]; print("check line sign: corr(spread_line, home margin) =", round(np.corrcoef(h.spread_line.fillna(0), h.home_points - h.away_points)[0, 1], 3))
