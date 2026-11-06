"""Watch the sim (J 2026-11-06): a few simulated games per game for the Bet Desk viewer (app/sim.js).
python3 models/sims/build.py [NBA CBB NFL CFB TENNIS]  -> app/data/sims_{nba,cbb,nfl,cfb,tennis}.json

Honesty rule: every replay is one simulated game from our model, never a forecast of the real game. Each sport plays games with
the model's own simulator (no new model here):
  NBA / CBB   possession by possession: the possession outcome shares (0-3 pts) of models/{nba,cbb}_ab/lib.py tilted to
              Model A's points per possession, possessions around Model A's pace, overtime on a tie.
  NFL         drive by drive: models/rebuild/football trace.py (same rules and tables as the tested NFL drive simulator),
              scoring odds solved so its means match Model A.
  CFB         drive by drive: models/cfb_ab/lib.py Sim (the engine behind CFB Model A's ranges), recorded step by step.
  TENNIS      point by point: serve-point odds, match-day form and pressure shifts from models/tennis_ab/lib.py.
For the four team sports a pool of games is played and the replays are drawn from it weighted to Model A's published spread
(margin and total, from its 10th-90th ranges), so over many replays the results match Model A's win % and ranges. Tennis is
drawn straight from the same Markov model Model A uses (no reweighting needed). Each file carries an audit block per game
(Model A vs the weighted pool) so a drift between the viewer and Model A shows up in the build log and the file.
Only games from today (Denver) through the next day (NFL/CFB: next 6 days) are kept, including games that already started."""
import os, sys, json, math, datetime as dt, numpy as np
from zoneinfo import ZoneInfo
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.normpath(f"{HERE}/../.."); APP = os.environ.get("SIMS_APP_DATA", f"{ROOT}/app/data")   # override for testing
Z = 2.5631                     # 10th-90th percentile width of a normal, in sds
TODAY = dt.datetime.now(ZoneInfo("America/Denver")).date()
K_KEEP = {"NBA": 6, "CBB": 5, "NFL": 6, "CFB": 6, "TENNIS": 5}
SPAN = {"NBA": 1, "CBB": 1, "NFL": 6, "CFB": 6, "TENNIS": 1}

def _mod(name, path):
    import importlib.util
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp)
    sys.path.insert(0, os.path.dirname(path)); sp.loader.exec_module(m); sys.path.pop(0); return m

def in_window(sp, g):
    d = g.get("d") or (g.get("k") or "")[:10]
    if not d: return True
    try: d = dt.date.fromisoformat(d[:10])
    except ValueError: return True
    return TODAY <= d <= TODAY + dt.timedelta(days=SPAN[sp])

def target(A):
    """Model A's published margin / total centre and spread (sd from the 10th-90th range)."""
    return (A["hs"] - A["as"], max((A["m90"] - A["m10"]) / Z, 1.0), A["hs"] + A["as"], max((A["t90"] - A["t10"]) / Z, 1.0))

def draw(H, A_, tgt, k, rng):
    """weights that turn the pool's (margin, total) spread into Model A's (normal ratio, so the pool's own shape, e.g. key
    numbers, stays); k replays drawn without replacement; audit = Model A vs weighted pool."""
    m, t = H - A_, H + A_; mm, sm, mt, st = tgt
    pm, ps, qm, qs = m.mean(), max(m.std(), 1e-6), t.mean(), max(t.std(), 1e-6)
    lw = -0.5 * (((m - mm) / sm) ** 2 - ((m - pm) / ps) ** 2 + ((t - mt) / st) ** 2 - ((t - qm) / qs) ** 2)
    w = np.exp(lw - lw.max()); w = np.minimum(w, 20 * w.mean()); w /= w.sum()
    idx = rng.choice(len(w), size=min(k, len(w)), replace=False, p=w)
    win = float((w * ((m > 0) + 0.5 * (m == 0))).sum())
    audit = dict(p_home_pool=round(win, 3), margin_pool=round(float((w * m).sum()), 1), total_pool=round(float((w * t).sum()), 1),
                 ess=int(1 / (w ** 2).sum()), n=len(w))
    return idx, audit

# ---------------------------------------------------------------- basketball: possession by possession
BB = {"NBA": dict(lib=f"{ROOT}/models/nba_ab/lib.py", reg=48, per=4, poss_sd=3.5),
      "CBB": dict(lib=f"{ROOT}/models/cbb_ab/lib.py", reg=40, per=2, poss_sd=3.0)}
def basketball(sp, G, rng, NP=3000):
    cfg = BB[sp]; L = _mod(f"{sp.lower()}_ab_lib", cfg["lib"])
    grid = np.round(np.arange(0.5, 1.701, 0.005), 3); Q = np.array([np.cumsum(L._tilt(x)) for x in grid])   # outcome cdf by ppp
    out = {}
    for gid, g in G.items():
        A = g.get("A") or {}
        if not A.get("poss") or A.get("m10") is None: continue
        hs, as_, poss = A["hs"], A["as"], A["poss"]; tgt = target(A)
        def play(eh, ea, n):
            N = np.clip(np.rint(rng.normal(poss, cfg["poss_sd"], n)), poss * 0.75, poss * 1.25).astype(int)
            ih = np.clip(np.rint((((hs + eh) / poss) - 0.5) / 0.005), 0, len(grid) - 1).astype(int)
            ia = np.clip(np.rint((((as_ + ea) / poss) - 0.5) / 0.005), 0, len(grid) - 1).astype(int)
            mx = int(N.max()); live = np.arange(mx)[None, :] < N[:, None]
            ch = (rng.random((n, mx))[:, :, None] > Q[ih][:, None, :]).sum(-1) * live
            ca = (rng.random((n, mx))[:, :, None] > Q[ia][:, None, :]).sum(-1) * live
            H, Aw = ch.sum(1), ca.sum(1); ot = np.zeros(n, int); oth, ota = [[] for _ in range(n)], [[] for _ in range(n)]
            mo = max(3, int(round(poss * 5 / cfg["reg"])))
            for _ in range(4):                                    # overtime periods (5 min) on a tie
                tie = np.flatnonzero(H == Aw)
                if not len(tie): break
                xh = (rng.random((len(tie), mo))[:, :, None] > Q[ih[tie]][:, None, :]).sum(-1)
                xa = (rng.random((len(tie), mo))[:, :, None] > Q[ia[tie]][:, None, :]).sum(-1)
                H[tie] += xh.sum(1); Aw[tie] += xa.sum(1); ot[tie] += 1
                for j, i in enumerate(tie): oth[i] += xh[j].tolist(); ota[i] += xa[j].tolist()
            return N, ch, ca, H, Aw, ot, oth, ota, mo
        # pilot: game luck alone; then add a rating / pace shock so the pool is a bit wider than Model A (reweighting narrows it)
        p = play(np.zeros(800), np.zeros(800), 800); m0, t0 = (p[3] - p[4]).std(), (p[3] + p[4]).std()
        em = math.sqrt(max(0.0, (1.15 * tgt[1]) ** 2 - m0 ** 2)); et = math.sqrt(max(0.0, (1.15 * tgt[3]) ** 2 - t0 ** 2))
        dm, dt_ = rng.normal(0, em, NP), rng.normal(0, et, NP)
        N, ch, ca, H, Aw, ot, oth, ota, mo = play((dt_ + dm) / 2, (dt_ - dm) / 2, NP)
        idx, audit = draw(H, Aw, tgt, K_KEEP[sp], rng)
        S = [{"o": int(rng.random() < 0.5), "n": int(N[i]), "ot": int(ot[i]), "mo": mo,
              "h": "".join(map(str, ch[i, :N[i]].tolist() + oth[i])), "a": "".join(map(str, ca[i, :N[i]].tolist() + ota[i])),
              "f": [int(H[i]), int(Aw[i])]} for i in idx]
        out[gid] = dict(h=g["h"], a=g["a"], d=g.get("d"), k=g.get("k"), A=_asum(A, ["p_ot", "poss"]), audit=audit, s=S)
    return out, dict(unit="possession", reg_min=cfg["reg"], periods=cfg["per"],
                     how=f"Possession by possession: each trip scores 0, 1, 2 or 3 with odds tilted to Model A's points per trip, about {{poss}} trips a side, 5-minute overtime on a tie.")

def _asum(A, extra=()):
    return {k: A[k] for k in ["hs", "as", "p_home", "m10", "m90", "t10", "t90", *extra] if A.get(k) is not None}

# ---------------------------------------------------------------- football: drive by drive (shared encoding)
# drive = [half, offense 0 home / 1 away, start yards from the end zone, result code, points, scorer 0/1, seconds]
# result codes: 0 TD, 1 FG, 2 punt, 3 downs / missed FG, 4 turnover, 5 return TD (defense scores), 6 safety, 7 end of half, 8 OT FG;
# labels per sport go in the file (NFL tables count downs and missed FGs as turnovers)
def nfl(G, rng, NP=800):
    import pandas as pd
    F = f"{ROOT}/models/rebuild/football"; sys.path.insert(0, F)
    import sim as S, trace as TR
    KW = dict(k_score=0.1, k_late=0.6, th_sd=0.3)    # build_drive_viewer.py / simtest settings
    R = {"Touchdown": 0, "Field goal": 1, "Punt": 2, "Turnover": 4, "Return TD": 5, "Safety": 6, "Field goal (OT)": 8}
    out = {}; ids = [k for k, g in G.items() if (g.get("A") or {}).get("m10") is not None]
    if not ids: return out, {}
    q, _ = S.simulate(pd.DataFrame({"pred_h": [G[k]["A"]["hs"] for k in ids], "pred_a": [G[k]["A"]["as"] for k in ids]}), nsim=2000, **KW)
    for j, gid in enumerate(ids):
        g, A = G[gid], G[gid]["A"]
        pool = [TR.trace_game(q.th_h.iat[j], q.th_a.iat[j], seed=int(rng.integers(1 << 30)), **KW) for _ in range(NP)]
        H = np.array([s["home"] for s in pool], float); Aw = np.array([s["away"] for s in pool], float)
        idx, audit = draw(H, Aw, target(A), K_KEEP["NFL"], rng)
        S_ = [{"f": [pool[i]["home"], pool[i]["away"]], "ot": pool[i]["ot"],
               "d": [[d["half"], 0 if d["side"] == "H" else 1, d["start"], R[d["res"]], d["pts"], 0 if d["who"] == "H" else 1,
                      int(round(d["dur"] * 60))] for d in pool[i]["drives"]]} for i in idx]
        out[gid] = dict(h=g["h"], a=g["a"], d=g.get("d"), k=g.get("k"), A=_asum(A, ["p3", "p7"]), audit=audit, s=S_)
    return out, dict(unit="drive", reg_min=60, periods=4, labels=["Touchdown", "Field goal", "Punt", "", "Turnover (incl. downs, missed FG)", "Return touchdown", "Safety", "End of half", "Overtime field goal"],
                     how="Drive by drive: each drive's result (touchdown, field goal, punt, turnover, return score, safety) is drawn from NFL drive tables by starting field position, with each team's scoring odds set so the average matches Model A; drive lengths and the next drive's start come from real drives. Overtime is simplified (first score wins).")

def cfb(G, rng, NP=1500):
    L = _mod("cfb_ab_lib", f"{ROOT}/models/cfb_ab/lib.py"); SIM = L.Sim(); SIM.rng = rng
    out = {}
    for gid, g in G.items():
        A = g.get("A") or {}
        if A.get("m10") is None: continue
        rec = []; H, Aw = SIM.run(A["hs"], A["as"], NP, rec=rec)
        idx, audit = draw(H, Aw, target(A), K_KEEP["CFB"], rng)
        S_ = []
        for i in idx:
            dr = []
            for r in rec:
                if not r["live"][i]: continue
                c, o = int(r["c"][i]), int(r["poss"][i]); ret = bool(r["ret"][i])
                code = 5 if ret else 7 if (r["done"][i] and c == 2) else {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6}[c]
                pts = int(r["off"][i] or r["dfn"][i]); who = o if r["off"][i] else 1 - o
                dr.append([r["half"], o, int(r["y"][i]), code, pts, who, int(round(float(r["sec"][i])))])
            S_.append({"f": [int(H[i]), int(Aw[i])], "ot": 0, "d": dr})
        out[gid] = dict(h=g["h"], a=g["a"], d=g.get("d"), k=g.get("k"), n=g.get("n"), A=_asum(A, ["p3", "p7"]), audit=audit, s=S_)
    return out, dict(unit="drive", reg_min=60, periods=4, labels=["Touchdown", "Field goal", "Punt", "Turnover on downs or missed FG", "Interception or fumble", "Return touchdown", "Safety", "End of half", ""],
                     how="Drive by drive with college drive tables from 2023-25 play-by-play (today's clock rules): touchdown and field-goal odds tilted so each team's average matches Model A, plus a shared game shock and a team shock (the CFB Model A simulator). The simulator does not play college overtime: a tie after four quarters stays a tie here.")

# ---------------------------------------------------------------- tennis: point by point
def tennis(M, rng, NP=300):
    T = _mod("tennis_ab_lib", f"{ROOT}/models/tennis_ab/lib.py")
    lg, sg = (lambda p: math.log(p / (1 - p))), (lambda x: 1 / (1 + math.exp(-x)))
    R = rng.random
    def game(p, a):                         # server's point odds p, pressure shifts a -> (server won, points as '1' server / '0')
        i = j = 0; s = []
        while True:
            x = lg(p)
            if i >= 3 and j >= 3 and i == j: x += a.get("dc", 0)
            elif j >= 3 and j > i: x += a.get("bp", 0)
            elif i >= 3 and i > j: x += a.get("gp", 0)
            elif i == 2 and j == 2: x += a.get("dc", 0)
            w = R() < sg(x); s.append(w); i += w; j += not w
            if i >= 4 and i - j >= 2: return True, s
            if j >= 4 and j - i >= 2: return False, s
    def one(pa, pb, bo, adj):
        need = (bo + 1) // 2; sv = int(R() < 0.5); first = sv; sets = []; pts = []; wa = wb = 0
        while wa < need and wb < need:
            ga = gb = 0
            while True:
                if ga == 6 and gb == 6:                       # tiebreak: first server serves 1, then 2 each
                    i = j = 0; k = 0
                    while not ((i >= 7 or j >= 7) and abs(i - j) >= 2):
                        s_ = sv if k % 4 in (0, 3) else 1 - sv
                        wonA = (R() < pa) if s_ == 0 else not (R() < pb)
                        pts.append(wonA); i += wonA; j += not wonA; k += 1
                    ga, gb = (7, 6) if i > j else (6, 7); sv = 1 - sv; break
                if sv == 0: w, s = game(pa, adj); pts += s; ga += w; gb += not w
                else: w, s = game(pb, adj); pts += [not x for x in s]; gb += w; ga += not w
                sv = 1 - sv
                if (ga >= 6 or gb >= 6) and abs(ga - gb) >= 2: break
            sets.append([ga, gb]); wa += ga > gb; wb += gb > ga
        return first, sets, pts
    out = {}
    for mm in M:
        A = mm.get("A") or {}
        if not A.get("serve_pts"): continue
        pa0, pb0 = A["serve_pts"]; mu, d = (lg(pa0) + lg(pb0)) / 2, lg(pa0) - lg(pb0); bo = A.get("best_of", 3); adj = T.ADJ.get(mm.get("sex", "m"), {})
        pool = []
        for _ in range(NP):
            dd = d + T.SD * rng.standard_normal(); pool.append(one(sg(mu + dd / 2), sg(mu - dd / 2), bo, adj))
        win = np.mean([sum(a > b for a, b in s) > sum(b > a for a, b in s) for _, s, _ in pool])
        tot = np.mean([sum(a + b for a, b in s) for _, s, _ in pool])
        key = str(mm["fid"])
        out[key] = dict(h=mm["p1"], a=mm["p2"], k=mm.get("start"), A={k: A[k] for k in ("win_p1", "tot_mean", "tot_range", "best_of") if k in A},
                        audit=dict(win_pool=round(float(win), 3), tot_pool=round(float(tot), 1), n=NP),
                        s=[{"f": f, "sets": s, "p": "".join("1" if x else "0" for x in p)} for f, s, p in pool[:K_KEEP["TENNIS"]]])
    return out, dict(unit="point", how="Point by point: each player's serve-point odds from Model A, a match-day form draw, tour pressure shifts at break, game and deuce points, tiebreak at 6-6 in every set.")

# ---------------------------------------------------------------- run
SRC = {"NBA": "nba_ab.json", "CBB": "cbb_ab.json", "NFL": "nfl_ab.json", "CFB": "cfb_ab.json", "TENNIS": "tennis_ab.json"}
def build(sp):
    f = f"{APP}/{SRC[sp]}"
    if not os.path.exists(f) or (sp == "TENNIS" and not os.path.exists(f"{ROOT}/models/tennis_ab/lib.py")):
        print(sp, "skipped:", SRC[sp], "or its model code is missing"); return
    J = json.load(open(f)); rng = np.random.default_rng(int(dt.datetime.now().strftime("%Y%m%d")) * 10 + len(sp))
    if sp == "TENNIS":
        games, meta = tennis([m for m in J.get("matches", []) if m.get("fid") and in_window(sp, {"k": m.get("start") or ""})], rng)   # the desk shows matches with an odds id only
    else:
        G = {k: g for k, g in (J.get("games") or {}).items() if in_window(sp, g)}
        games, meta = basketball(sp, G, rng) if sp in BB else nfl(G, rng) if sp == "NFL" else cfb(G, rng)
    doc = {"made": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "sport": sp, "model_a_made": J.get("made"),
           "note": "Simulated games from our model, not a prediction of the real game. Each replay is one game the simulator played; over many replays they match Model A's win chance and ranges.",
           **meta, "games": games}
    o = f"{APP}/sims_{sp.lower()}.json"; json.dump(doc, open(o, "w"), separators=(",", ":"))
    bad = []      # audit: the replays' pool vs Model A (win % beyond 3 standard errors of the pool, or the average margin 3+ pts off)
    for gid, g in games.items():
        a, A = g["audit"], g["A"]
        p, pa = (A.get("win_p1"), a["win_pool"]) if sp == "TENNIS" else (A["p_home"], a["p_home_pool"])
        n = a["n"] if sp == "TENNIS" else a["ess"]; se = math.sqrt(max(p * (1 - p), 0.01) / n)
        if abs(pa - p) > 3 * se + 0.01: bad.append(f'{g["a"]} @ {g["h"]} win {pa:.3f} vs {p:.3f}')
        if sp != "TENNIS" and abs(a["margin_pool"] - (A["hs"] - A["as"])) >= 3: bad.append(f'{g["a"]} @ {g["h"]} margin {a["margin_pool"]} vs {A["hs"] - A["as"]:.1f}')
    print(f"{sp}: {len(games)} games -> {os.path.basename(o)} {os.path.getsize(o) // 1024} KB; audit: " + (f"{len(bad)} off Model A: {bad}" if bad else "every game's replays match Model A"), flush=True)

if __name__ == "__main__":
    for sp in (sys.argv[1:] or list(SRC)): build(sp.upper())
