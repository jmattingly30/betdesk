"""CBB v2 for Bet Desk (J 2026-10-27: the new version is the main CBB model; today's models/cbb becomes the shadow).
CBB v2 = today's models/cbb projection stacked with the rebuild's full stack (preseason roster prior + continuing
absences) using cbb_gap weights (data/cbb_v2_coef.json, fit on lined games 2013-2019; tested 2020-2026).
For every upcoming slate game: rebuild projection (ratings walked through every past game, then next season's start),
today's projection (models/cbb predict.price), and the stacked v2 score. -> app/data/cbb_v2.json
python3 live_cbb.py [slate.json]"""
import sys, os, json, datetime as dt, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import engine as E, ctx, m_prior
ROOT = os.path.normpath(f"{HERE}/../../..")
CBB = f"{ROOT}/models/cbb"
FIT_TO = 2016

def clip_new_season(season):
    """The coming season's roster features (models/cbb prep_roster_2027.py, real rosters from CollegeBasketballData) are
    measured from full rosters, so transfer minutes run above anything the prior was fit on; keep them inside the range
    of past seasons."""
    Z = m_prior.Z
    if season not in Z.index.get_level_values(0): return 0
    past = Z[Z.index.get_level_values(0) < season]
    Z.loc[season] = Z.loc[season].clip(past.min(), past.max(), axis=1).values
    return int((Z.index.get_level_values(0) == season).sum())

def rebuild_state():
    tg, nc = ctx.load_tg("cbb"); prm = ctx.stack_params("cbb", nc)
    g = E.games_from_teamgames(tg)
    start = m_prior.prior_hook(tg, prm, FIT_TO=FIT_TO); hooks = {"season_start": start}
    p0 = E.run(tg, prm, hooks=hooks)[0]
    names, Xh, Xa, Fh, Fa = ctx.build(tg, g, ["avail"])
    m = (g.season <= FIT_TO).values; mu = np.vstack([Fh[m], Fa[m]]).mean(0)
    B = ctx.fit(g, p0, Fh - mu, Fa - mu, FIT_TO)
    ph, pa = (Xh - mu) @ B, (Xa - mu) @ B
    sh = {gid: (ph[i], pa[i], 0 * ph[i], 0 * pa[i]) for i, gid in enumerate(g.game_id)}
    _, R = E.run(tg, prm, shifts=sh, hooks=hooks)
    conf = tg.sort_values("date").groupby("team").conf.last().to_dict()
    return R, start, int(g.season.max()), g.date.min(), (np.zeros(len(mu)) - mu) @ B, conf

def main():
    slate_p = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/app/data/game_slates.json"
    games = json.load(open(slate_p))["CBB"]["games"]
    co = json.load(open(f"{HERE}/data/cbb_v2_coef.json"))
    cwd = os.getcwd(); os.chdir(CBB); sys.path.insert(0, CBB)
    import predict as PR
    rs = PR.preseason_ratings(); os.chdir(cwd)
    R, start, last_season, day0, sh0, conf = rebuild_state()
    season = last_season + 1
    print("next-season roster rows:", clip_new_season(season))
    R.new_season(); start(R, season)   # next season's start (roster prior applies when its features exist)
    out = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "season": season, "coef": co,
           "note": "hs/as = CBB v2 (main model). today_hs/today_as = previous CBB model (shadow).", "games": {}}
    miss = []
    for x in games:
        try: h, a = PR.find(rs, x["h"]), PR.find(rs, x["a"])
        except SystemExit as e: miss.append(f'{x["a"]} @ {x["h"]}: {e}'); continue
        hid, aid = int(h.team_id), int(a.team_id)
        if hid not in R.idx or aid not in R.idx: miss.append(f'{x["a"]} @ {x["h"]}: no rebuild history'); continue
        day = (pd.Timestamp(x["d"]) - day0).days; R.age(R.idx[hid], day); R.age(R.idx[aid], day)
        n = bool(x.get("n"))
        lh, la, poss, _, _ = R.predict(hid, aid, n, R.cidx(conf.get(hid, -1)), R.cidx(conf.get(aid, -1)))
        rh = E.expit(lh + sh0); ra = E.expit(la + sh0)   # avail shift with no known absences (as in the backtest for most games)
        rb_h = float(E.ppp({c: rh[k] for k, c in enumerate(E.COMP)}) * poss); rb_a = float(E.ppp({c: ra[k] for k, c in enumerate(E.COMP)}) * poss)
        t = PR.price(rs, h["name"], a["name"], neutral=n)
        tm, tt = t["margin"], t["total"]; rm, rt = rb_h - rb_a, rb_h + rb_a
        cm, ct = co["margin"], co["total"]
        m = cm["const"] + cm["tm"] * tm + cm["rm"] * rm + cm["x_av"] * 0.0
        tot = ct["const"] + ct["tt"] * tt + ct["rt"] * rt + ct["poss_hat"] * float(poss)
        w1 = co.get("week1")
        if w1 and 0 <= (pd.Timestamp(x["d"]) - pd.Timestamp(w1["season_start"])).days < w1["days"]: tot += w1["total_offset"]   # opening-week totals fix
        out["games"][x["id"]] = {"h": x["h"], "a": x["a"], "d": x["d"], "n": int(n), "hs": round((tot + m) / 2, 2), "as": round((tot - m) / 2, 2),
                                 "today_hs": t["home_pts"], "today_as": t["away_pts"], "rb_hs": round(rb_h, 2), "rb_as": round(rb_a, 2), "poss": round(float(poss), 1)}
    json.dump(out, open(f"{ROOT}/app/data/cbb_v2.json", "w"), separators=(",", ":"))
    print(len(out["games"]), "games ->", f"{ROOT}/app/data/cbb_v2.json", "| unmatched:", len(miss)); [print("  ", s) for s in miss[:10]]
    for k, v in list(out["games"].items())[:8]: print(v["a"], "@", v["h"], "v2", v["hs"], v["as"], "| today", v["today_hs"], v["today_as"], "| rebuild", v["rb_hs"], v["rb_as"])

if __name__ == "__main__": main()
