"""NHL team-games 2010-11 .. now from sportsdataverse-data play-by-play (nhl_pbp_full) + schedules.
Per team-game: goals by state (even, power play, short-handed, empty net), expected goals (pbp xg) and unblocked
attempts by state, shots on goal, power-play and even-strength time, penalties taken, starting goalie, OT/shootout.
Usage: python3 prep_pbp.py <download_dir> [seasons]  -> data/teamgames_nhl.parquet"""
import sys, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); dl = sys.argv[1]
seasons = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(2011, 2028)
COLS = ["event_type", "event_team_type", "home_skaters", "away_skaters", "empty_net", "xg", "period", "period_type",
        "game_id", "game_seconds", "event_goalie_id", "home_goalie_id", "away_goalie_id", "home_abbr", "away_abbr", "season_type"]
out = []
for y in seasons:
    sch = pd.read_parquet(f"{dl}/nhl_schedule_{y}.parquet")
    sch = sch[sch.game_type.isin(["R", "P"]) & sch.game_state.isin(["OFF", "FINAL"])]
    # some pbp files are labelled one season off: read the neighbours and keep this schedule's game ids
    parts = []
    for f in (y - 1, y, y + 1):
        p = f"{dl}/pbp_{f}.parquet"
        if os.path.exists(p):
            x = pd.read_parquet(p, columns=COLS); parts.append(x[x.game_id.isin(sch.game_id.astype(float))])
    d = pd.concat(parts).drop_duplicates()
    d["game_id"] = d.game_id.astype("int64")
    d = d.sort_values(["game_id", "game_seconds"], kind="stable")
    reg = d[d.period_type != "SHOOTOUT"]
    # time in each manpower state (home perspective), from the gap to the next event
    nxt = reg.groupby("game_id").game_seconds.shift(-1)
    dur = (nxt - reg.game_seconds).clip(lower=0, upper=600).fillna(0)
    hs, as_ = reg.home_skaters.fillna(5), reg.away_skaters.fillna(5)
    t = pd.DataFrame({"game_id": reg.game_id, "pp_h": dur * (hs > as_), "pp_a": dur * (as_ > hs), "ev": dur * (hs == as_)})
    T = t.groupby("game_id").sum()
    sh = reg[reg.event_type.isin(["SHOT", "GOAL", "MISSED_SHOT"])].copy()
    own = np.where(sh.event_team_type == "home", sh.home_skaters, sh.away_skaters)
    opp = np.where(sh.event_team_type == "home", sh.away_skaters, sh.home_skaters)
    en = sh.empty_net.fillna(False).astype(bool)
    sh["st"] = np.select([en, own > opp, own < opp], ["en", "pp", "sh"], "ev")
    sh["g"] = (sh.event_type == "GOAL").astype(int); sh["sog"] = sh.event_type.isin(["SHOT", "GOAL"]).astype(int); sh["f"] = 1
    sh["xg"] = sh.xg.fillna(sh.groupby("st").xg.transform("mean"))
    A = sh.pivot_table(index=["game_id", "event_team_type"], columns="st", values=["g", "xg", "f"], aggfunc="sum", fill_value=0)
    A.columns = [f"{a}_{b}" for a, b in A.columns]
    S = sh.groupby(["game_id", "event_team_type"]).sog.sum()
    pen = d[d.event_type == "PENALTY"].groupby(["game_id", "event_team_type"]).size()
    # starting goalie = the one who faced the most shots for that side
    gk = sh.dropna(subset=["event_goalie_id"]).groupby(["game_id", "event_team_type", "event_goalie_id"]).size().reset_index(name="n")
    gk = gk.sort_values("n").groupby(["game_id", "event_team_type"]).tail(1).set_index(["game_id", "event_team_type"]).event_goalie_id
    per = d.groupby("game_id").agg(maxp=("period", "max"), so=("period_type", lambda s: (s == "SHOOTOUT").any()))
    sog_so = d[(d.period_type == "SHOOTOUT") & (d.event_type == "GOAL")].groupby(["game_id", "event_team_type"]).size()
    sch = sch.set_index("game_id")
    for side, oth in (("home", "away"), ("away", "home")):
        gid = sch.index.values
        r = pd.DataFrame({"game_id": gid, "season": y, "date": pd.to_datetime(sch.game_date.values), "playoff": (sch.game_type == "P").values,
                          "team": sch[f"{side}_team_abbr"].values, "opp": sch[f"{oth}_team_abbr"].values, "ha": side,
                          "gf": sch[f"{side}_score"].values, "ga": sch[f"{oth}_score"].values})
        for c in A.columns:
            r[c] = A[c].reindex(list(zip(gid, [side] * len(gid)))).fillna(0).values
            r["o_" + c] = A[c].reindex(list(zip(gid, [oth] * len(gid)))).fillna(0).values
        r["sog"] = S.reindex(list(zip(gid, [side] * len(gid)))).fillna(0).values
        r["o_sog"] = S.reindex(list(zip(gid, [oth] * len(gid)))).fillna(0).values
        r["pens"] = pen.reindex(list(zip(gid, [side] * len(gid)))).fillna(0).values
        r["o_pens"] = pen.reindex(list(zip(gid, [oth] * len(gid)))).fillna(0).values
        r["toi_pp"] = T[f"pp_{side[0]}"].reindex(gid).values; r["toi_sh"] = T[f"pp_{oth[0]}"].reindex(gid).values; r["toi_ev"] = T.ev.reindex(gid).values
        # goalie this team's shooters faced = opponent's goalie; own goalie faced the other side's shots
        r["goalie"] = gk.reindex(list(zip(gid, [oth] * len(gid)))).values
        r["o_goalie"] = gk.reindex(list(zip(gid, [side] * len(gid)))).values
        r["ot"] = per.maxp.reindex(gid).values > 3; r["so"] = per.so.reindex(gid).fillna(False).values
        # final score from the play-by-play (old schedule scores are unreliable): non-shootout goals + 1 for a shootout win
        own_so = sog_so.reindex(list(zip(gid, [side] * len(gid)))).fillna(0).values
        opp_so = sog_so.reindex(list(zip(gid, [oth] * len(gid)))).fillna(0).values
        r["g_reg"] = r[[c for c in ("g_ev", "g_pp", "g_sh", "g_en") if c in r]].sum(axis=1)
        r["o_g_reg"] = r[[c for c in ("o_g_ev", "o_g_pp", "o_g_sh", "o_g_en") if c in r]].sum(axis=1)
        r["gf"] = r.g_reg + (r.so & (own_so > opp_so)); r["ga"] = r.o_g_reg + (r.so & (opp_so > own_so))
        out.append(r)
    print(y, len(sch), "games; pbp games", d.game_id.nunique(), flush=True)
tg = pd.concat(out, ignore_index=True)
tg = tg[tg.toi_ev.notna()]
tg.to_parquet(os.environ.get("OUT", f"{HERE}/data/teamgames_nhl.parquet"))
print(tg.groupby("season").size().to_dict())
print(tg[["gf", "g_ev", "g_pp", "g_en", "xg_ev", "xg_pp", "sog", "toi_pp", "toi_ev", "pens"]].describe().T[["mean", "std"]])
