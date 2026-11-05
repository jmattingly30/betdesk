"""NBA full stack (lineup + schedule) for UPCOMING games, for the shadow run.
The lineup known before an upcoming game follows the backtest's 'known' rule: players from the team's last game plus
regulars who missed only that game are expected to play (latest state); regulars out two or more games running are out. 'Full' adds back regulars from the last five games who are missing.
No injury feed reaches the cloud, so news since the last game is not seen.
project(slate) -> {game_id: (home_pts, away_pts)}; needs data/live/{current_players,lineup_agg}.parquet fresh
(live_lineup.py) and data/teamgames_nba.parquet."""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("LINEUP_AGG", f"{HERE}/data/live/lineup_agg.parquet")
import engine as E, ctx

STATS = ['pts', 'fga', 'fta', 'fg3a', 'fg3m', 'ast', 'tov', 'oreb', 'dreb', 'stl', 'blk', 'pf', 'rapm']
RC = ['r_' + s for s in STATS]

def agg(df):
    """Same rotation weighting as live_lineup.py: ten-man rotation (more if 11+ project 15+ minutes)."""
    df = df.nlargest(max(10, int((df.min_proj >= 15).sum())), 'min_proj')
    w = df.min_proj.to_numpy(); w = w / w.sum() * 5
    return pd.Series(np.r_[w @ df[RC].to_numpy(), len(df)], index=RC + ['n'])

def future_lineups(cur, tg, team, gid):
    tt = tg[tg.team == team]; tdates = np.sort(tt[tt.season == tt.season.max()].date.unique())   # this season only
    if not len(tdates): return None
    last = tdates[-1]; win = tdates[-5:]
    c = cur[cur.team == team]
    regs = c[(c.date < last) & (c.date >= win[0]) & (c.min_proj >= 12)]
    # as the backtest's 'known' lineup: a regular who missed only the last game (rest, a one-game knock) is expected
    # back; one who has missed two or more in a row is out
    back = regs[regs.date == tdates[-2]] if len(tdates) > 1 else regs.iloc[:0]
    known = pd.concat([c[c.date == last], back]); regs = regs.drop(back.index)
    if len(known) < 5: return None
    k = agg(known); f = agg(pd.concat([known, regs]))
    row = {"game_id": gid, "team": team, **{x: k[x] for x in RC}, "n": k["n"],
           **{x + "_full": f[x] for x in RC}, "n_full": f["n"], **{x + "_known": k[x] for x in RC}, "n_known": k["n"],
           "miss_min": float(regs.min_proj.sum()), "miss_names": ";".join(regs.name.astype(str)), "miss_min_new": 0.0}
    return row

def project(slate):
    mods, pf = ctx.STACK["nba"]
    prm = {**E.DEFAULT, **json.load(open(f"{HERE}/{pf}"))}
    tg = pd.read_parquet(f"{HERE}/data/teamgames_nba.parquet")
    cur = pd.read_parquet(f"{HERE}/data/live/current_players.parquet"); cur["date"] = pd.to_datetime(cur.date)
    L = pd.read_parquet(os.environ["LINEUP_AGG"])
    g = E.games_from_teamgames(tg)
    season = int(g.season.max())
    fut_tg, fut_g, fut_L = [], [], []
    for i, gm in enumerate(slate):
        gid = -(i + 1); d = pd.Timestamp(gm["d"])   # negative ids: never clash with real game ids
        rows = [future_lineups(cur, tg, t, gid) for t in (gm["h"], gm["a"])]
        if any(r is None for r in rows): continue
        fut_L += rows
        fut_tg += [{"game_id": gid, "team": gm["h"], "date": d, "season": season, "ha": "home"},
                   {"game_id": gid, "team": gm["a"], "date": d, "season": season, "ha": "away"}]
        fut_g.append({"game_id": gid, "team": gm["h"], "team_a": gm["a"], "season": season, "slate": gm})
    if not fut_g: return {}
    # features for history + upcoming games through the same transforms as the backtest
    tg_x = pd.concat([tg[["game_id", "team", "date", "season", "ha"]], pd.DataFrame(fut_tg)], ignore_index=True)
    g_x = pd.concat([g[["game_id", "team", "team_a", "season"]], pd.DataFrame(fut_g).drop(columns="slate")], ignore_index=True)
    pd.concat([L, pd.DataFrame(fut_L)], ignore_index=True).to_parquet(f"{HERE}/data/live/lineup_agg_future.parquet")
    os.environ["LINEUP_AGG"] = f"{HERE}/data/live/lineup_agg_future.parquet"
    try:
        names, Xh, Xa, Fh, Fa = ctx.build(tg_x, g_x, mods)
    finally:
        os.environ["LINEUP_AGG"] = f"{HERE}/data/live/lineup_agg.parquet"
    m = (g_x.season <= 2017).values
    mu = np.vstack([Fh[m], Fa[m]]).mean(0); Xh, Xa, Fh, Fa = Xh - mu, Xa - mu, Fh - mu, Fa - mu
    nh = len(g)
    p0 = E.run(tg, prm)[0]
    B = ctx.fit(g, p0, Fh[:nh], Fa[:nh], 2017)
    ph, pa = Xh @ B, Xa @ B
    sh = {gid: (ph[i], pa[i], 0 * ph[i], 0 * pa[i]) for i, gid in enumerate(g.game_id)}
    _, R = E.run(tg, prm, shifts=sh)
    day0 = g.date.min(); out = {}
    for i, fg in enumerate(fut_g):
        j = nh + i; gm = fg["slate"]; h, a = gm["h"], gm["a"]
        d = (pd.Timestamp(gm["d"]) - day0).days; R.age(R.idx[h], d); R.age(R.idx[a], d)
        lh, la, poss, _, _ = R.predict(h, a, bool(gm.get("n")))
        rh, ra = E.expit(lh + ph[j]), E.expit(la + pa[j])
        out[gm["id"]] = (float(E.ppp({c: rh[k] for k, c in enumerate(E.COMP)}) * poss),
                         float(E.ppp({c: ra[k] for k, c in enumerate(E.COMP)}) * poss),
                         fut_L[2 * i]["miss_names"], fut_L[2 * i + 1]["miss_names"])
    return out

if __name__ == "__main__":
    ROOT = os.path.normpath(f"{HERE}/../../..")
    slate = json.load(open(f"{ROOT}/app/data/game_slates.json"))["NBA"]["games"]
    for k, v in project(slate).items(): print(k, round(v[0], 1), round(v[1], 1), "| out h:", v[2] or "-", "| out a:", v[3] or "-")
