"""Shadow run of the rebuilt engines on live games (graded only; nothing feeds Bet Desk).
python3 shadow.py [nba] [nfl] [nhl] [cbb]
CBB is the reverse: Bet Desk runs CBB v2 and the previous models/cbb model is the shadow (see cbb()).
  1. refresh: NBA team-games from the project's player box table; NFL drives from the newest nflverse play-by-play
  2. project every upcoming game (NBA: Bet Desk slate; NFL: nflverse schedule, this week + next) and keep the
     last projection made before kickoff in history.json
  3. grade finished games: rebuild vs today's Bet Desk model (grading/history/GAMES.json.gz, unblended `hs`/`as`)
     vs the market line, on the same games -> report.json
NBA: full stack (lineup + schedule) when basketball/live_lineup.py has been run since the last game, else ratings only;
NFL uses ratings + depth-chart starting QB + missing regulars (injury report) + dome (live_nfl.py); weather not wired."""
import sys, os, json, gzip, datetime as dt, subprocess, urllib.request
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RB = os.path.dirname(HERE); ROOT = os.path.normpath(f"{RB}/../..")
sys.path.insert(0, RB)
HIST = f"{HERE}/history.json"; REPORT = f"{HERE}/report.json"
SP = os.environ.get("DL", "/tmp/rebuild_dl")
now = dt.datetime.now(dt.timezone.utc)
hist = json.load(open(HIST)) if os.path.exists(HIST) else {}

def load(name, path):
    import importlib.util
    sp_ = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp_)
    sys.path.insert(0, os.path.dirname(path)); sp_.loader.exec_module(m); sys.path.pop(0); return m

def keep(sport, gid, kick, row):
    """Store a projection while the game has not started (last pre-kickoff projection wins)."""
    k = f"{sport}|{gid}"
    if kick and pd.Timestamp(kick).tz_convert("UTC") <= now and k in hist: return
    hist[k] = {**row, "sp": sport, "id": str(gid), "k": kick, "made": now.strftime("%Y-%m-%d %H:%M UTC")}

def nba():
    E = load("bb_engine", f"{RB}/basketball/engine.py"); prep = load("bb_prep", f"{RB}/basketball/prep.py")
    tg = prep.finish(prep.nba()); tg.to_parquet(f"{RB}/basketball/data/teamgames_nba.parquet")
    prm = {**E.DEFAULT, **json.load(open(f"{RB}/basketball/data/tuned_stack_nba.json"))}
    p, R = E.run(tg, prm); day0 = E.games_from_teamgames(tg).date.min()
    slate = json.load(open(f"{ROOT}/app/data/game_slates.json"))["NBA"]["games"]; n = 0
    # full stack (lineup + schedule) when the rebuild's lineup feed is current (basketball/live_lineup.py, ~8 min)
    cp = pd.read_parquet(f"{RB}/basketball/data/live/current_players.parquet")
    full = {}
    if pd.to_datetime(cp.date).max() >= tg.date.max():
        full = load("bb_live", f"{RB}/basketball/live_future_nba.py").project(slate)
    for gm in slate:
        h, a = gm["h"], gm["a"]
        if h not in R.idx or a not in R.idx: continue
        if gm["id"] in full:
            ph, pa = full[gm["id"]][:2]
            keep("NBA", gm["id"], gm["k"], {"h": h, "a": a, "d": gm["d"], "hs": round(ph, 2), "as": round(pa, 2), "src": "stack", "ln": gm.get("ln", {}).get("L"), "lt": gm.get("ln", {}).get("T")}); n += 1
            continue
        d = (pd.Timestamp(gm["d"]) - day0).days; R.age(R.idx[h], d); R.age(R.idx[a], d)
        lh, la, poss, _, _ = R.predict(h, a, bool(gm.get("n")))
        rh, ra = E.expit(lh), E.expit(la)
        ph = float(E.ppp({c: rh[k] for k, c in enumerate(E.COMP)}) * poss); pa = float(E.ppp({c: ra[k] for k, c in enumerate(E.COMP)}) * poss)
        keep("NBA", gm["id"], gm["k"], {"h": h, "a": a, "d": gm["d"], "hs": round(ph, 2), "as": round(pa, 2), "ln": gm.get("ln", {}).get("L"), "lt": gm.get("ln", {}).get("T")}); n += 1
    # finals for grading (ESPN ids, same as the slate)
    fin = tg.pivot_table(index="game_id", columns="ha", values="pts", aggfunc="first")
    return n, {str(g): (r.get("home"), r.get("away")) for g, r in fin.iterrows()}

def nfl():
    os.makedirs(f"{SP}/nfl", exist_ok=True); B = "https://github.com/nflverse/nflverse-data/releases/download"
    for y in range(2006, 2027):
        f = f"{SP}/nfl/pbp_{y}.parquet"
        if y == 2026 or not os.path.exists(f): urllib.request.urlretrieve(f"{B}/pbp/play_by_play_{y}.parquet", f)
    urllib.request.urlretrieve("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", f"{SP}/nfl/games.csv")
    subprocess.run([sys.executable, f"{RB}/football/prep_drives.py", f"{SP}/nfl"], check=True, capture_output=True)
    subprocess.run([sys.executable, f"{RB}/football/prep_qb.py", f"{SP}/nfl"], check=True, capture_output=True)
    E = load("fb_engine", f"{RB}/football/engine.py")
    T = pd.read_parquet(f"{RB}/football/data/teamgames_nfl.parquet")
    p, R = E.run(T, json.load(open(f"{RB}/football/data/tuned_nfl.json")))
    G = pd.read_csv(f"{SP}/nfl/games.csv"); G = G[G.season == 2026]
    up = G[G.result.isna()].sort_values("gameday"); wk = up.week.min(); up = up[up.week <= wk + 1]
    # live inputs (live_nfl.py): depth-chart QB1 and missing regulars from snaps + this week's injury report
    for f in ("depth_charts/depth_charts_2026", "injuries/injuries_2026", "snap_counts/snap_counts_2026"):
        urllib.request.urlretrieve(f"{B}/{f}.parquet", f"{SP}/nfl/" + ("snaps_2026" if "snap" in f else f.split("/")[1]) + ".parquet")
    for y in range(2013, 2026):
        if not os.path.exists(f"{SP}/nfl/snaps_{y}.parquet"): urllib.request.urlretrieve(f"{B}/snap_counts/snap_counts_{y}.parquet", f"{SP}/nfl/snaps_{y}.parquet")
    LN = load("live_nfl", f"{RB}/football/live_nfl.py")
    dc = pd.read_parquet(f"{SP}/nfl/depth_charts_2026.parquet"); dc["dt"] = pd.to_datetime(dc.dt)
    dc = dc[dc.pos_abb.eq("QB") & dc.pos_rank.eq(1)].sort_values("dt").groupby("team").tail(1).set_index("team").gsis_id.to_dict()
    fo = f"{RB}/../team-strength/nfl/qb_overrides.json"   # manual starting-QB overrides (J 2026-11-05, BAL Brissett)
    if os.path.exists(fo):
        today = pd.Timestamp.now(tz="America/Denver").strftime("%Y-%m-%d")
        dc.update({t: o["id"] for t, o in json.load(open(fo)).get("teams", {}).items() if o.get("until", "9999") >= today})
    up = up[up.home_team.isin(R.idx) & up.away_team.isin(R.idx)].reset_index(drop=True)
    U = pd.DataFrame({"home": up.home_team, "away": up.away_team, "h_qb": up.home_team.map(dc), "a_qb": up.away_team.map(dc)})
    qsh, _ = LN.qb_terms(U)
    inj = pd.read_parquet(f"{SP}/nfl/injuries_2026.parquet"); ish = np.zeros((len(up), 2))
    for wk_, ix in up.groupby("week").groups.items():
        ish[ix] = LN.inj_terms(U.loc[ix], f"{SP}/nfl", inj[inj.week == wk_])[0]
    dome_mu = T.roof.isin(["dome", "closed"]).mean(); n = 0
    for i, r in enumerate(up.itertuples()):
        lh, la = R.predict(r.home_team, r.away_team, r.location == "Neutral")
        ph, pa = E.points(lh, R.ret) + R.lvl, E.points(la, R.ret) + R.lvl
        dm = 0.754 * ((r.roof in ("dome", "closed")) - dome_mu); ph += dm; pa += dm
        ph += qsh[i][0] + ish[i][0]; pa += qsh[i][1] + ish[i][1]
        kick = pd.Timestamp(f"{r.gameday} {r.gametime or '13:00'}", tz="America/New_York").tz_convert("UTC").isoformat()
        keep("NFL", r.game_id, kick, {"h": r.home_team, "a": r.away_team, "d": r.gameday, "hs": round(float(ph), 2), "as": round(float(pa), 2),
                                      "ln": None if pd.isna(r.spread_line) else -float(r.spread_line), "lt": None if pd.isna(r.total_line) else float(r.total_line)}); n += 1
    done = G[G.result.notna()]
    return n, {g: (h, a) for g, h, a in zip(done.game_id, done.home_score, done.away_score)}

def nhl():
    """NHL: refresh this season from play-by-play, frame ratings with the usual starting goalie (goalie used for the
    margin only, as kept), schedule terms (back-to-back, 3-in-4, rest); lineup term not wired for future games."""
    os.makedirs(f"{SP}/nhl", exist_ok=True); B = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
    urllib.request.urlretrieve(f"{B}/nhl_pbp_full/play_by_play_2027.parquet", f"{SP}/nhl/pbp_2027.parquet")
    urllib.request.urlretrieve(f"{B}/nhl_schedules/nhl_schedule_2027.parquet", f"{SP}/nhl/nhl_schedule_2027.parquet")
    tmp = f"{SP}/nhl/tg_2027.parquet"
    subprocess.run([sys.executable, f"{RB}/hockey/prep_pbp.py", f"{SP}/nhl", "2027"], check=True, capture_output=True, env={**os.environ, "OUT": tmp})
    path = f"{RB}/hockey/data/teamgames_nhl.parquet"
    tg = pd.concat([pd.read_parquet(path).query("season < 2027"), pd.read_parquet(tmp)], ignore_index=True); tg.to_parquet(path)
    E = load("hk_engine", f"{RB}/hockey/engine.py")
    prm = json.load(open(f"{RB}/hockey/data/tuned_nhl.json"))
    _, Rg = E.run(tg, {**prm, "gk_mode": "usual"}); _, Rn = E.run(tg, {**prm, "gk_mode": "none"})
    L = E.league(tg).iloc[-1]
    sch = pd.read_parquet(f"{SP}/nhl/nhl_schedule_2027.parquet")
    up = sch[~sch.game_state.isin(["OFF", "FINAL"]) & sch.game_type.eq("R")].copy()
    up["date"] = pd.to_datetime(up.game_date); up = up[up.date <= up.date.min() + pd.Timedelta(days=3)]
    # schedule features (as m_sched.py: own and opponent b2b / rest3 / 3-in-4, centred on 2012-2021)
    hist_d = pd.concat([tg[["team", "date"]], up.rename(columns={"home_team_abbr": "team"})[["team", "date"]],
                        up.rename(columns={"away_team_abbr": "team"})[["team", "date"]]]).drop_duplicates().sort_values("date")
    def feats(team, date):
        ds = hist_d[(hist_d.team == team) & (hist_d.date < date)].date
        rest = (date - ds.iloc[-1]).days if len(ds) else 9; g2 = (date - ds.iloc[-2]).days if len(ds) > 1 else 9
        return np.array([rest == 1, rest >= 3, g2 <= 3], float)
    bo = np.array([-0.0709, 0.0187, -0.0563]); bd = np.array([0.0732, -0.0165, 0.0258]); fmu = np.array([0.17, 0.30, 0.13])
    usual = {}
    gk = tg.dropna(subset=["goalie"]).sort_values("date")
    for t_, g_ in gk.groupby("opp"): usual[t_] = g_.goalie.tail(10).mode().iat[0]   # 'goalie' = the goalie the team's shooters faced
    n = 0
    for r in up.itertuples():
        h, a = r.home_team_abbr, r.away_team_abbr
        if h not in Rg.idx or a not in Rg.idx: continue
        out = {}
        for nm, R in (("g", Rg), ("n", Rn)):
            lh, la, gka, gkh = R.predict(h, a, False, usual.get(a) if nm == "g" else None, usual.get(h) if nm == "g" else None)
            for side, l, g_ in (("h", lh, gka), ("a", la, gkh)):
                evx, ppt, ppx, fin = np.exp(l[0]), np.exp(l[1]), np.exp(l[2]), np.exp(l[3] + g_)
                out[nm + side] = evx * L.ev_min / 60 * fin + ppt / 60 * ppx * fin + L.sh + L.en + L.otg * 0.5
        m = ((out["gh"] - out["nh"]) - (out["ga"] - out["na"])) / 2      # goalie: margin only
        fh, fa = feats(h, r.date), feats(a, r.date)
        sh = (fh - fmu) @ bo + (fa - fmu) @ bd; sa = (fa - fmu) @ bo + (fh - fmu) @ bd
        keep("NHL", r.game_id, r.game_time, {"h": h, "a": a, "d": r.game_date, "hs": round(float(out["nh"] + m + sh), 3), "as": round(float(out["na"] - m + sa), 3), "ln": None, "lt": None}); n += 1
    done = sch[sch.game_state.isin(["OFF", "FINAL"])]
    return n, {str(g): (hh, aa) for g, hh, aa in zip(done.game_id, done.home_score, done.away_score)}

def cbb():
    """CBB (J 2026-10-27): CBB v2 (models/cbb stacked with the rebuild, basketball/live_cbb.py) is Bet Desk's main CBB
    model; the previous models/cbb model is the one shadow-graded here (today_hs/today_as in app/data/cbb_v2.json).
    In the report, "rebuild" = this shadow model and "today" = Bet Desk's graded snapshot (CBB v2 once the desk uses it).
    Finals from ESPN's 2027 men's college basketball schedule (sportsdataverse)."""
    v = json.load(open(f"{ROOT}/app/data/cbb_v2.json"))
    slate = {g["id"]: g for g in json.load(open(f"{ROOT}/app/data/game_slates.json"))["CBB"]["games"]}
    n = 0
    for gid, x in v["games"].items():
        s = slate.get(gid, {}); ln = s.get("ln") or {}
        keep("CBB", gid, s.get("k"), {"h": x["h"], "a": x["a"], "d": x["d"], "hs": x["today_hs"], "as": x["today_as"], "v2": [x["hs"], x["as"]],
                                      "model": "previous models/cbb (shadow)", "ln": ln.get("L"), "lt": ln.get("T")}); n += 1
    fin = {}
    try:
        S = pd.read_parquet("https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/mbb_schedule_2027.parquet",
                            columns=["id", "status_type_completed", "home_score", "away_score"])
        for r in S[S.status_type_completed == True].itertuples(): fin[str(r.id)] = (float(r.home_score), float(r.away_score))
    except Exception as e: print("CBB finals unavailable:", e)
    return n, fin

def grade(finals):
    today = json.load(gzip.open(f"{ROOT}/models/grading/history/GAMES.json.gz"))
    rows = []
    for k, v in hist.items():
        sp, gid = v["sp"], v["id"]; f = finals.get(sp, {}).get(gid)
        if not f or f[0] is None or pd.isna(f[0]): continue
        t = today.get(k, {})
        rows.append({"sp": sp, "id": gid, "d": v["d"], "ah": float(f[0]), "aa": float(f[1]), "rh": v["hs"], "ra": v["as"],
                     "th": t.get("hs"), "ta": t.get("as"), "ln": v.get("ln"), "lt": v.get("lt")})
    out = {"made": now.strftime("%Y-%m-%d %H:%M UTC"), "sports": {}}
    if not rows: return out
    D = pd.DataFrame(rows)
    for sp, d in D.groupby("sp"):
        m = d.ah - d.aa; t = d.ah + d.aa; both = d.th.notna()
        def err(h, a, mask):
            x = d[mask]; return {"n": int(mask.sum()), "team_mae": round(float(((x[h] - x.ah).abs() + (x[a] - x.aa).abs()).mean() / 2), 2),
                                 "margin_mae": round(float(((x[h] - x[a]) - (x.ah - x.aa)).abs().mean()), 2),
                                 "total_mae": round(float(((x[h] + x[a]) - (x.ah + x.aa)).abs().mean()), 2)}
        res = {"rebuild_all": err("rh", "ra", pd.Series(True, index=d.index)), "rebuild_same_games": err("rh", "ra", both),
               "today_same_games": err("th", "ta", both) if both.any() else None}
        ln = d.ln.notna() & both
        if ln.any():
            x = d[ln]; res["line_same_games"] = {"n": int(ln.sum()), "margin_mae": round(float((x.ln + (x.ah - x.aa)).abs().mean()), 2),
                                                 "total_mae": round(float((x["lt"] - (x.ah + x.aa)).abs().mean()), 2)}
        out["sports"][sp] = res
    return out

if __name__ == "__main__":
    want = [a.upper() for a in sys.argv[1:]] or ["NBA", "NFL", "NHL", "CBB"]; finals = {}
    for sp in want:
        n, fin = {"NBA": nba, "NFL": nfl, "NHL": nhl, "CBB": cbb}[sp](); finals[sp] = fin; print(sp, "projected", n, "games", flush=True)
    json.dump(hist, open(HIST, "w"), indent=0)
    rep = grade(finals); json.dump(rep, open(REPORT, "w"), indent=1); print(json.dumps(rep, indent=1))
