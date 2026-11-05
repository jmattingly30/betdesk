"""One team-game table per sport with the box counts the possession engine needs.
CBB: models/cbb/data (hoopR team box, D1 games). NBA: summed from models/nba-player player box (2015-2027)."""
import os, numpy as np, pandas as pd
M = "/mnt/project-files/sports-analytics/models"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"); os.makedirs(OUT, exist_ok=True)
COLS = ["pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta", "orb", "drb", "tov"]

def cbb():
    t = pd.read_parquet(f"{M}/cbb/data/cbb_team_games.parquet")
    g = pd.read_parquet(f"{M}/cbb/data/cbb_games.parquet")[["game_id", "d1", "neutral", "home", "away", "home_score", "away_score"]]
    t = t[t.game_id.isin(g.loc[g.d1, "game_id"])]
    t = t.rename(columns={"o_fgm": "o_fgm"})
    keep = ["game_id", "season", "date", "team", "opp", "ha", "neutral", "season_type"] + COLS + ["o_" + c for c in COLS]
    t = t[[c for c in keep if c in t.columns]].copy()
    t["playoff"] = (t.season_type == 3)
    return t

def nba():
    p = pd.read_parquet(f"{M}/nba-player/data/player_games.parquet")
    p = p[p.played == True] if "played" in p else p
    agg = p.groupby(["game_id", "season", "date", "team", "opp", "is_home", "playoff"], as_index=False)[
        ["pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "tov"]].sum()
    agg.columns = ["game_id", "season", "date", "team", "opp", "is_home", "playoff"] + COLS
    sc = p.groupby(["game_id", "team"]).team_score.first().rename("team_score")
    agg = agg.join(sc, on=["game_id", "team"])
    # team turnovers not credited to a player are missing from player sums; small, noted
    opp = agg[["game_id", "team"] + COLS].rename(columns={"team": "opp", **{c: "o_" + c for c in COLS}})
    t = agg.merge(opp, on=["game_id", "opp"])
    t["ha"] = np.where(t.is_home, "home", "away"); t["neutral"] = False
    bad = (t.pts - t.team_score).abs() > 0
    print("NBA team-games where player points don't sum to the team score:", int(bad.sum()))
    t = t[~bad].drop(columns=["is_home", "team_score"])
    both = t.groupby("game_id").team.transform("size") == 2
    return t[both]

def finish(t):
    t["date"] = pd.to_datetime(t.date).dt.tz_localize(None).dt.normalize()
    t["poss"] = 0.5 * ((t.fga - t.orb + t.tov + 0.44 * t.fta) + (t.o_fga - t.o_orb + t.o_tov + 0.44 * t.o_fta))
    return t.sort_values(["date", "game_id", "ha"]).reset_index(drop=True)

if __name__ == "__main__":
    for name, f in (("cbb", cbb), ("nba", nba)):
        t = finish(f()); t.to_parquet(f"{OUT}/teamgames_{name}.parquet")
        print(name, t.shape, t.season.min(), t.season.max(), "ppp", round((t.pts / t.poss).mean(), 3), "poss", round(t.poss.mean(), 1))
