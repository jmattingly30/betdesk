"""NHL team-games 2010-11 .. now from sportsdataverse-data (schedules + team box scores).
Box scores before 2023-24 carry no game id: they are in game order, so each away/home pair is matched to the
schedule (regular season + playoffs, by game id) on teams and goals.
Usage: python3 prep.py <download_dir>  -> data/teamgames_nhl.parquet (one row per team-game)."""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); dl = sys.argv[1]
rows = []
for y in range(2011, 2028):
    s = pd.read_parquet(f"{dl}/nhl_schedule_{y}.parquet"); t = pd.read_parquet(f"{dl}/team_box_{y}.parquet")
    s = s[s.game_type.isin(["R", "P"]) & s.game_state.isin(["OFF", "FINAL"])].sort_values("game_id").reset_index(drop=True)
    if "game_id" in t:
        a = t[t.home_away == "away"].set_index("game_id"); h = t[t.home_away == "home"].set_index("game_id")
        ok = s.game_id.isin(a.index) & s.game_id.isin(h.index); g = s[ok]
        A, Hh = a.loc[g.game_id].reset_index(), h.loc[g.game_id].reset_index()
    else:
        a = t.iloc[0::2].reset_index(drop=True); h = t.iloc[1::2].reset_index(drop=True)
        key = list(zip(s.away_team_abbr, s.home_team_abbr, s.away_score, s.home_score))
        used, gi, ai, hi = set(), [], [], []
        j = 0
        for i in range(len(a)):
            k = (a.team_abbrev[i], h.team_abbrev[i], a.goals[i], h.goals[i])
            for jj in range(j, min(j + 40, len(key))):
                if jj not in used and key[jj] == k:
                    used.add(jj); gi.append(jj); ai.append(i); j = max(j, jj - 5); break
        g = s.iloc[gi]; A = a.iloc[ai].reset_index(drop=True); Hh = h.iloc[ai].reset_index(drop=True)
        print(y, "matched", len(gi), "of", len(a), "box pairs;", len(s), "games", flush=True)
    g = g.reset_index(drop=True)
    for side, X, O in (("home", Hh, A), ("away", A, Hh)):
        rows.append(pd.DataFrame({"game_id": g.game_id.values, "season": y, "date": pd.to_datetime(g.game_date.values),
            "playoff": (g.game_type == "P").values, "team": X.team_abbrev.values, "opp": O.team_abbrev.values, "ha": side,
            "gf": X.goals.values, "ga": O.goals.values, "sf": X.shots_on_goal.values, "sa": O.shots_on_goal.values,
            "ppg": X.power_play_goals.values, "pim": X.pim.values, "o_pim": O.pim.values, "o_ppg": O.power_play_goals.values,
            "sv": X.saves.values, "o_sv": O.saves.values}))
tg = pd.concat(rows, ignore_index=True)
# shootout winners get +1 goal in the final score; flag games decided past regulation from the linescore-less data later
tg.to_parquet(f"{HERE}/data/teamgames_nhl.parquet")
print(tg.groupby("season").size().to_dict()); print(tg[["gf", "sf", "ppg"]].describe().T)
