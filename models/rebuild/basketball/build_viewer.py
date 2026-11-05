"""Data for the simulation viewer: tonight's NBA games from the rebuilt engine's current ratings.
Writes viewer/nba_sims.json: per game the projection, 2,000 simulated games (score/margin/total distributions,
win %, overtime %, scoring-run stats) and 30 full possession-by-possession sample games."""
import sys, os, json, datetime as dt, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import engine as E, simtrace as T
prm = {**E.DEFAULT, **json.load(open(f"{HERE}/data/tuned_stack_nba.json"))}
cal = json.load(open(f"{HERE}/data/simcal_nba.json"))
tg = pd.read_parquet(f"{HERE}/data/teamgames_nba.parquet")
p, R = E.run(tg, prm)
slate = json.load(open(f"{HERE}/../../../app/data/game_slates.json"))["NBA"]["games"]
day = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
games = [g for g in slate if g["d"] == day]
day0 = E.games_from_teamgames(tg).date.min()
NS, NT = 2000, 12
out = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "date": day, "sport": "NBA", "nsim": NS,
       "note": "Team ratings from the rebuilt possession engine (four factors, Kalman). Lineup and schedule adjustments are not applied to live games yet (lineup feed for 2026-27 not built).",
       "games": []}
for gm in games:
    h, a = gm["h"], gm["a"]
    if h not in R.idx or a not in R.idx: continue
    d = (pd.Timestamp(gm["d"]) - day0).days
    R.age(R.idx[h], d); R.age(R.idx[a], d)
    lh, la, poss, vh, va = R.predict(h, a, bool(gm.get("n")))
    rh, ra = E.expit(lh), E.expit(la)
    ph = float(E.ppp({c: rh[k] for k, c in enumerate(E.COMP)}) * poss); pa = float(E.ppp({c: ra[k] for k, c in enumerate(E.COMP)}) * poss)
    kw = dict(poss_sd=cal["poss_sd"], wob=cal["wob"], k_score=cal["k_score"], w_shared=cal["w_shared"])
    # centre the simulator on the projection: score effects pull margins in, so lift each side's shooting logits
    # until simulated means match the formula means (two short passes)
    sh_h = sh_a = 0.0
    for it in range(6):
        bh = rh.copy(); ba = ra.copy()
        for x, d_ in ((bh, sh_h), (ba, sh_a)):
            x[3] = E.expit(E.logit(x[3]) + d_); x[4] = E.expit(E.logit(x[4]) + d_)
        if it == 5: break
        ss = [T.trace_game(bh, ba, poss, seed=10_000 + s, **kw) for s in range(600)]
        mh = np.mean([q["home"] for q in ss]); ma = np.mean([q["away"] for q in ss])
        sh_h += (ph - mh) / (poss * 0.95); sh_a += (pa - ma) / (poss * 0.95)
    sims = [T.trace_game(bh, ba, poss, seed=s, **kw) for s in range(NS)]
    H_ = np.array([s["home"] for s in sims]); A_ = np.array([s["away"] for s in sims]); OT = np.array([s["ot"] for s in sims])
    M, Tt = H_ - A_, H_ + A_
    # biggest lead each way and lead changes, per sim
    big_h, big_a, chg = [], [], []
    for s in sims:
        lead = np.array([q["h"] - q["a"] for q in s["poss"]])
        big_h.append(int(max(0, lead.max()))); big_a.append(int(max(0, -lead.min())))
        sg = np.sign(lead[lead != 0]); chg.append(int((sg[1:] != sg[:-1]).sum()))
    def hist(x):
        v, c = np.unique(x, return_counts=True); return [[int(i), int(j)] for i, j in zip(v, c)]
    ln = gm.get("ln") or {}
    out["games"].append({
        "id": gm["id"], "kick": gm["k"], "home": h, "away": a,
        "proj": {"home": round(ph, 1), "away": round(pa, 1), "poss": round(float(poss), 1),
                 "margin": round(ph - pa, 1), "total": round(ph + pa, 1)},
        "rates_h": [round(float(v), 4) for v in rh], "rates_a": [round(float(v), 4) for v in ra], "comp": E.COMP,
        "sim": {"home_mean": round(float(H_.mean()), 1), "away_mean": round(float(A_.mean()), 1),
                "p_home": round(float((M > 0).mean()), 3), "p_ot": round(float((OT > 0).mean()), 3),
                "sd_margin": round(float(M.std()), 2), "sd_total": round(float(Tt.std()), 2),
                "q": {k: [float(np.percentile(x, z)) for z in (10, 25, 50, 75, 90)] for k, x in (("home", H_), ("away", A_), ("margin", M), ("total", Tt))},
                "hist_home": hist(H_), "hist_away": hist(A_), "hist_margin": hist(M), "hist_total": hist(Tt),
                "joint": [[int(x), int(y)] for x, y in zip(H_[:600], A_[:600])],
                "lead_changes_mean": round(float(np.mean(chg)), 1), "big_lead_home_mean": round(float(np.mean(big_h)), 1),
                "big_lead_away_mean": round(float(np.mean(big_a)), 1)},
        "line": {"spread": ln.get("L"), "total": ln.get("T")},
        "samples": sims[:NT]})
    print(h, a, out["games"][-1]["proj"], out["games"][-1]["sim"]["p_home"], flush=True)
os.makedirs(f"{HERE}/viewer", exist_ok=True)
json.dump(out, open(f"{HERE}/viewer/nba_sims.json", "w"), separators=(",", ":"))
print("wrote", len(out["games"]), "games", os.path.getsize(f"{HERE}/viewer/nba_sims.json") // 1024, "KB")
