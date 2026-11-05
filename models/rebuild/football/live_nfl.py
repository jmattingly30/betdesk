"""Live NFL inputs for upcoming games (shadow run): tonight's expected starting QB and missing regulars.
Same features and fits as the kept backtest modules (m_qb.py on frame1, m_inj.py unit mode on frame1qw); only the
'who plays tonight' part differs, because the backtest knew it and a future game does not:
  starting QB      nflverse depth chart QB1 (ESPN), latest snapshot. (Bet Desk's qb_change flags use the usual
                   starter and missed Jalen Hurts' injury in 2026 week 7.)
  missing regulars a regular (50%+ of his unit's snaps in 3 of the team's last 5 games) is out if he is Out or
                   Doubtful on this week's injury report, or he missed the team's last game and is not on this
                   week's report at all (injured reserve and similar). Questionable = plays.
qb_terms(up) and inj_terms(up, week) return point shifts (home, away) per row of up (columns home, away, h_qb, a_qb)."""
import os, glob, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
GRP = {"T": "OL", "G": "OL", "C": "OL", "OL": "OL", "WR": "WR", "TE": "TE", "RB": "RB", "FB": "RB",
       "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
       "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB"}
OFF = ["OL", "WR", "TE", "RB"]

def _ridge(Xh, Xa, p, fit, lam):
    XX = np.vstack([Xh[fit], Xa[fit]]); yy = np.concatenate([(p.act_h - p.pred_h)[fit], (p.act_a - p.pred_a)[fit]])
    mu = XX.mean(0); Z = XX - mu; sd = Z.std(0) + 1e-9
    beta = np.linalg.solve((Z / sd).T @ (Z / sd) + lam * np.eye(Z.shape[1]), (Z / sd).T @ (yy - yy.mean())) / sd
    return mu, beta

def qb_terms(up, HL=24.0, K=250.0, B0=-0.08):
    Q = pd.read_parquet(f"{HERE}/data/qb_games_nfl.parquet")
    T = pd.read_parquet(f"{HERE}/data/teamgames_nfl.parquet")
    T["starter"] = np.where(T.is_home == 1, T.home_qb_id, T.away_qb_id); T = T.sort_values(["date", "game_id"])
    lg = Q.merge(T[["game_id", "season"]].drop_duplicates(), on="game_id").groupby("season").apply(lambda d: d.epa.sum() / d.db.sum())
    a = 0.5 ** (1 / HL); a_t = 0.5 ** (1 / 6); S = {}; team_ew = {}; last = {}
    val = lambda q: (S[q][0] + B0 * K) / (S[q][1] + K) if q in S else B0
    Qg = {k: v for k, v in Q.groupby("game_id")}; rows = []
    for gid, g in T.groupby("game_id", sort=False):
        f = {}
        for r in g.itertuples():
            v = val(r.starter); f[r.is_home] = (v - team_ew.get(r.team, v), float(last.get(r.team) not in (None, r.starter)))
        if len(f) == 2: rows.append({"game_id": gid, "h_dq": f[1][0], "h_new": f[1][1], "a_dq": f[0][0], "a_new": f[0][1]})
        for r in g.itertuples():
            v = val(r.starter); team_ew[r.team] = a_t * team_ew.get(r.team, v) + (1 - a_t) * v; last[r.team] = r.starter
        qg = Qg.get(gid); s_lg = lg.get(g.season.iloc[0], 0)
        if qg is not None:
            for r in qg.itertuples():
                s = S.setdefault(r.qb, [0.0, 0.0]); s[0] = a * s[0] + (r.epa - s_lg * r.db); s[1] = a * s[1] + r.db
    F = pd.DataFrame(rows).set_index("game_id")
    p = pd.read_parquet(f"{HERE}/data/frame1_nfl.parquet").join(F, on="game_id").fillna(0)
    X = lambda o: np.column_stack([p[o + "dq"], p[o + "new"]])
    mu, beta = _ridge(X("h_"), X("a_"), p, ((p.season >= 2008) & (p.season <= 2017)).values, 20.0)
    out = []; info = []
    for r in up.itertuples():
        sh = []
        for team, qb in ((r.home, r.h_qb), (r.away, r.a_qb)):
            if qb is None or qb != qb: sh.append(0.0); continue
            v = val(qb); x = np.array([v - team_ew.get(team, v), float(last.get(team) not in (None, qb))])
            sh.append(float((x - mu) @ beta)); info.append((team, qb, round(v, 3), round(team_ew.get(team, v), 3), round(sh[-1], 2)))
        out.append(sh)
    return np.array(out), info

def inj_terms(up, snaps_dir, inj):
    """inj: this week's nflverse injury report rows (team, full_name, report_status)."""
    s = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{snaps_dir}/snaps_*.parquet"))])
    s = s[s.season >= 2013]; s["grp"] = s.position.map(GRP); s = s[s.grp.notna()]
    s["pct"] = np.where(s.grp.isin(OFF), s.offense_pct, s.defense_pct)
    gd = pd.read_csv(f"{snaps_dir}/games.csv")[["game_id", "gameday"]]; s = s.merge(gd, on="game_id").sort_values(["gameday", "game_id"])
    hist = {}; rows = []; names = {}
    for (gid, team), g in s.groupby(["game_id", "team"], sort=False):
        h = hist.get(team, []); today = set(g.pfr_player_id[g.pct > 0])
        rows.append({"game_id": gid, "team": team, **_missing(h, today)})
        hist[team] = (h + [dict(zip(g.pfr_player_id, zip(g.pct, g.grp)))])[-5:]
        names.update(zip(g.pfr_player_id, g.player))
    F = pd.DataFrame(rows); F["OFF"] = F[OFF].sum(1); F["DEF"] = F[["DL", "LB", "DB"]].sum(1)
    p = pd.read_parquet(f"{HERE}/data/frame1qw_nfl.parquet")
    fh = F.set_index(["game_id", "team"])[["OFF", "DEF"]]
    H_ = fh.reindex(pd.MultiIndex.from_frame(p[["game_id", "home"]])).fillna(0).values; A_ = fh.reindex(pd.MultiIndex.from_frame(p[["game_id", "away"]])).fillna(0).values
    Xh = np.column_stack([H_[:, 0], A_[:, 1]]); Xa = np.column_stack([A_[:, 0], H_[:, 1]])
    mu, beta = _ridge(Xh, Xa, p, ((p.season >= 2013) & (p.season <= 2017)).values, 30.0)
    status = {(r.team, r.full_name): r.report_status for r in inj.itertuples()}
    listed = {(r.team, r.full_name) for r in inj.itertuples()}
    def tonight(team):
        h = hist.get(team, [])
        if not h: return {"OFF": 0.0, "DEF": 0.0}, []
        lastg = h[-1]; out_ids = set(); who = []
        for pid in {p_ for gm in h for p_ in gm}:
            nm = names.get(pid); st = status.get((team, nm))
            if st in ("Out", "Doubtful") or (lastg.get(pid, (0,))[0] <= 0 and (team, nm) not in listed): out_ids.add(pid)
        play = {pid for gm in h for pid in gm} - out_ids
        m = _missing(h, play); who = [names.get(pid) for pid in _regulars(h) if pid in out_ids]
        return {"OFF": sum(m[k] for k in OFF), "DEF": m["DL"] + m["LB"] + m["DB"]}, who
    out = []; info = []
    for r in up.itertuples():
        th, wh = tonight(r.home); ta, wa = tonight(r.away)
        xh = np.array([th["OFF"], ta["DEF"]]); xa = np.array([ta["OFF"], th["DEF"]])
        out.append([float((xh - mu) @ beta), float((xa - mu) @ beta)]); info.append((r.home, wh, r.away, wa))
    return np.array(out), info, beta

def _regulars(h):
    cnt = {}
    for gm in h[-5:]:
        for pid, (pc, gr) in gm.items():
            if pc >= 0.5: cnt[pid] = cnt.get(pid, 0) + 1
    return [pid for pid, c in cnt.items() if c >= 3] if len(h) >= 3 else []

def _missing(h, today):
    f = {k: 0.0 for k in ["OL", "WR", "TE", "RB", "DL", "LB", "DB"]}
    if len(h) < 3: return f
    pctsum = {}; grp = {}
    for gm in h[-5:]:
        for pid, (pc, gr) in gm.items(): pctsum[pid] = pctsum.get(pid, 0) + pc; grp[pid] = gr
    for pid in _regulars(h):
        if pid not in today: f[grp[pid]] += pctsum[pid] / min(len(h), 5)
    return f
