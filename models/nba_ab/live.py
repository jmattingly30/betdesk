"""NBA Model A + Model B for Bet Desk (J 2026-11-05 rollout; third sport on the two-model system after CBB and NFL).
Model A = projected scores: the rebuild stack (Kalman team ratings + tonight's projected lineup + schedule, as shadow.py;
          ratings only when the lineup feed is stale) PLUS the component fix (comp_test.py / model_a_test.json 'final': our
          3P% and FT-rate terms were overweighted, OREB / 3PA share underweighted; team miss vs actual 9.228 -> 9.208 on 2019-26,
          2026 DK opener 9.284 -> 9.253), and the possession sim around those means (blowout easing, overtime) for win %,
          80% ranges and the overtime chance (sim spread calibrated to Model A's measured miss: margin sd 13.7, total 18.7).
Model B = bets built on Model A: sharp-weighted devigged market number (devig.WEIGHT, capped 1.5x, books 3+ pts off the median
          line dropped), the share of our gap that has been real at the opener (confidence fit, model_b_test.json 'live':
          engines agree, lineup news, games played, back-to-backs, favorite size, model split), line timing (fresh = posted in
          the last 24 h and tip 8+ h away, opener-like; later = close-like, where NBA gaps lost at every size), EV at every
          Colorado book, tier A = bet, B = lean (tracked, not a bet), C = pass. Player props are tiered on the desk (PROPS).
Run after shadow.py nba and export_slates.py:  python3 models/nba_ab/live.py  -> app/data/nba_ab.json (+ history/runs.jsonl)"""
import json, os, sys, math, datetime as dt, importlib.util, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/../model-b")
from lib import sim_game, ncdf, comp_fix, COMP
import devig as DV
import ml_anchor as MLA
ROOT = os.path.normpath(f"{HERE}/../.."); APP = f"{ROOT}/app/data"; BB = f"{ROOT}/models/rebuild/basketball"
MY_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'espnbet', 'fanatics', 'betrivers', 'ballybet', 'bet365', 'hardrockbet', 'circasports']   # J: every CO-legal book
NAMES = dict(pinnacle='Pinnacle', betonlineag='BetOnline', draftkings='DraftKings', fanduel='FanDuel', betmgm='BetMGM', williamhill_us='Caesars',
             espnbet='ESPN BET', fanatics='Fanatics', betrivers='BetRivers', ballybet='Bally Bet', hardrockbet='Hard Rock', bet365='bet365', circasports='Circa')
TA = json.load(open(f"{HERE}/model_a_test.json")); TB = json.load(open(f"{HERE}/model_b_test.json"))
SDM, SDT, FIX = TA["sd_margin"], TA["sd_total"], TA["final"]
LV = TB["live"]; Q90 = LV["q90"]
FRESH_H, TIP_H = 24, 8      # fresh (opener-like) = this book posted the line in the last 24 h AND tip is 8+ h away (untested split; NBA openers post ~22 h before tip)
B_LATE, B_TOT_OPEN = 0.0, 0.30   # share of a gap trusted on a close-like line (research b .04 spreads, -.06 totals) / on a fresh total (b .30)
NOW = dt.datetime.now(dt.timezone.utc)
dec = DV.dec_from_am
# PROPS (desk applies these in redesign.js, pfAll wrapper; model-b-confidence props.json NBA_live_open = desk projections vs OddsPapi 2026 openers, 5%+ EV):
PROPS = {"A": "Proven (DK/FD overs, 12%+ EV on our model alone: 201 bets +11.6% ROI) and other overs at 5%+ EV on points, threes, assists, PRA "
              "(threes 58.6% +9.0%, points 55.8% +4.9%, assists 55.1% +3.0% at the opener)",
         "B": "overs on rebounds and turnovers (rebounds lost at 5%+ EV: 48.8% -7.9% open, -8.0% close): Lean, tracked, not a bet",
         "C": "blocks and steals (blocks 41% -26%, steals -11% at the opener) dropped; unders held back (no edge, CLV negative)"}

def inv(p):
    a = 0.147; y = 2 * p - 1; s = math.copysign(1, y); l = math.log(1 - y * y); t = 2 / (math.pi * a) + l / 2
    return math.sqrt(2) * s * math.sqrt(math.sqrt(t * t - l / a) - t)
def hours_since(ts, now=None):
    if not ts: return None
    try: return ((now or NOW) - dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 3600
    except ValueError: return None
def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def market(bk, kind):
    """sharp-weighted devigged market number (home margin 'm' or total 't'); books 3+ pts off the median line dropped"""
    i = 0 if kind == "m" else 3
    pts = [r[i] for r in bk.values() if r[i] is not None and r[i + 1] and r[i + 2]]
    if not pts: return None, []
    med = float(np.median(pts)); acc = tw = 0.0; used = []
    for b, r in bk.items():
        pt, p1, p2 = r[i], r[i + 1], r[i + 2]
        if pt is None or not p1 or not p2 or abs(pt - med) >= 3: continue
        p = min(max(DV.power([dec(p1), dec(p2)])[0], 0.02), 0.98)
        mu = (-pt + SDM * inv(p)) if kind == "m" else (pt + SDT * inv(p))
        w = min(DV.WEIGHT.get(b, 0.8), DV.MAX_RATIO); acc += w * mu; tw += w; used.append(b)
    return (acc / tw, used) if tw else (None, [])

def best_price(bk, po, side_home, kind, mu, tip_h):
    out = None; sd = SDM if kind == "m" else SDT
    for b in MY_BOOKS:
        r = bk.get(b)
        if not r: continue
        pt, ph, pa = (r[0], r[1], r[2]) if kind == "m" else (r[3], r[4], r[5])
        if pt is None or not ph or not pa: continue
        if kind == "m": p_h = ncdf((mu + pt) / sd); p, pr, lab = (p_h, ph, pt) if side_home else (1 - p_h, pa, -pt)
        else: p_o = 1 - ncdf((pt - mu) / sd); p, pr, lab = (p_o, ph, pt) if side_home else (1 - p_o, pa, pt)
        ev = p * dec(pr) - 1; age = hours_since((po.get(b) or [None] * 3)[0 if kind == "m" else 1])
        if out is None or ev > out["ev"]:
            out = dict(book=b, book_name=NAMES.get(b, b), pt=lab, price=pr, p=round(p, 4), ev=round(ev, 4), age_h=None if age is None else round(age, 1))
    if out: out["fresh"] = out["age_h"] is not None and out["age_h"] <= FRESH_H and tip_h >= TIP_H
    return out

def desk_margin(N, h, a, neu=False):
    """the desk NBA team model's own margin (bet-desk.html renderNba, full rosters, no line): the second engine for 'engines agree'"""
    if not N or h not in N["teams"] or a not in N["teams"]: return None
    def lineup(t):
        r = sorted([p for p in N["teams"][t]["players"] if p[1] > 0], key=lambda p: -p[1]); r = r[:max(10, sum(1 for p in r if p[1] >= 15))]
        tot = sum(p[1] for p in r); x = np.zeros(len(N["wo"]))
        for p in r: x += (p[1] / tot * 240) / 48 * np.array(p[2:2 + len(x)], float)
        return x
    H, A = N["teams"][h], N["teams"][a]; xh, xa = lineup(h), lineup(a)
    luh = xh @ N["wo"] + xa @ N["wd"] - N["centre"]; lua = xa @ N["wo"] + xh @ N["wd"] - N["centre"]
    poss = H["tempo"] + A["tempo"] - H["T0"]; hca = 0 if neu else H["hca"]
    sh = (H["mu"] + H["O"] - A["D"] + hca + N["k"] * luh) * poss / 100; sa = (H["mu"] + A["O"] - H["D"] - hca + N["k"] * lua) * poss / 100
    return float(sh - sa)

def conf_share(feat):
    c = LV["coef"]; s = c["edge"]
    for f in LV["features"]: s += c[f] * (feat[f] - LV["mu"][f]) / LV["sd"][f]
    return s

def reasons(feat, fresh, age, tip_h, miss):
    R = []
    R.append(["+" if feat["engines_same_side"] else "-", "The desk's NBA team model is on the same side of the line" if feat["engines_same_side"] else "The desk's NBA team model is on the other side of the line (gaps lost when the engines split)"])
    if feat["lineup_drives"]: R.append(["+", f"Tonight's projected lineup moves Model A {feat['ls_abs']:.1f} pts toward this side (lineup news is where NBA opener gaps were real)"])
    elif feat["ls_abs"] >= 1: R.append(["·", f"Projected lineup moves Model A {feat['ls_abs']:.1f} pts, toward the other side"])
    if miss: R.append(["·", "Not in tonight's projected lineup (from recent games played, not the injury report): " + "; ".join(miss)])
    g = int(feat["gp_min"])
    R.append(["+" if feat["gp_mid"] else "-" if g <= 5 else "·", f"Fewest games played by either team: {g}" + (" (16-40 games in has been the best window)" if feat["gp_mid"] else " (5 or fewer: early-season gaps lost 46%, so tier A waits)" if g <= 5 else "")])
    if feat["b2b_any"]: R.append(["+", "A team is on a back-to-back"])
    if feat["fav_size"] >= 10.5: R.append(["-", f"Big favorite ({feat['fav_size']:.1f}): gaps at 10.5+ have held up least"])
    if feat["model_split"] >= 4: R.append(["-", f"Model A and the desk model are {feat['model_split']:.1f} pts apart"])
    if age is None: R.append(["·", "No post time for this price"])
    elif fresh: R.append(["+", f"Line posted {age:.0f} h ago, tip in {tip_h:.0f} h: opener-like, where our NBA gaps have been real (tier A 57-62% out of sample)"])
    else: R.append(["-", f"Line up {age:.0f} h, tip in {tip_h:.0f} h: close-like, where NBA spread gaps lost at every size"])
    return R

def main():
    slate = json.load(open(f"{APP}/game_slates.json"))["NBA"]["games"]
    desk = json.load(open(f"{APP}/games_models.json")).get("NBA")
    E = load("bb_engine", f"{BB}/engine.py")
    tg = pd.read_parquet(f"{BB}/data/teamgames_nba.parquet"); prm = {**E.DEFAULT, **json.load(open(f"{BB}/data/tuned_stack_nba.json"))}
    p0, R = E.run(tg, prm); day0 = E.games_from_teamgames(tg).date.min()
    upc = [g for g in slate if g.get("k") and hours_since(g["k"]) is not None and hours_since(g["k"]) < 0]
    cp = pd.read_parquet(f"{BB}/data/live/current_players.parquet"); full = {}
    if pd.to_datetime(cp.date).max() >= tg.date.max():
        cwd = os.getcwd(); os.chdir(BB); os.environ.setdefault("LINEUP_AGG", f"{BB}/data/live/lineup_agg.parquet"); sys.path.insert(0, BB); sys.path.insert(0, os.path.dirname(BB))
        try: full = load("bb_live", f"{BB}/live_future_nba.py").project(upc)
        finally: os.chdir(cwd)
    season = int(tg.season.max()); cur = tg[tg.season == season]
    gp = cur.groupby("team").size().to_dict(); last = tg.groupby("team").date.max().to_dict()
    out = {"made": NOW.strftime("%Y-%m-%d %H:%M UTC"), "sport": "NBA", "data_through": str(tg.date.max())[:10], "lineup_feed": bool(full),
           "how": {"A": f"Rebuild stack (ratings + projected lineup + schedule) + component fix (3P% and FT rate down-weighted) + possession sim (margin sd {SDM}, total sd {SDT})",
                   "B": f"Sharp-weighted devigged market number; share of our gap trusted from the opener confidence fit; fresh = posted <= {FRESH_H} h and tip {TIP_H}+ h away. "
                        "Tier A = bet, B = lean (tracked, not a bet), C = pass."},
           "tested": {"A_close_2019_26": TA["yardstick_close_2019_2026"]["fix"], "A_base_close_2019_26": TA["yardstick_close_2019_2026"]["base"],
                      "A_dk_open_2026": TA["yardstick_dk_open_2026"]["fix"], "B_spread_sbr_21_22": TB["sbr_fit18_20_test21_22"]["rules"]["R4_R1_no_early"],
                      "B_spread_dk_2026": TB["sbr_all_to_dk2026"]["rules"]["R4_R1_no_early"], "B_total_dk_2026": TB["totals_dk_open_2026"]["raw"], "B_total_sbr_21_22": TB["totals_sbr_open_2021_22"]["raw"]},
           "props": PROPS, "games": {}}
    tot_gaps = []; recs = {}
    for gm in sorted(upc, key=lambda g: g["k"]):
        h, a, gid = gm["h"], gm["a"], str(gm["id"])
        if h not in R.idx or a not in R.idx: continue
        d = (pd.Timestamp(gm["d"]) - day0).days; R.age(R.idx[h], d); R.age(R.idx[a], d)
        lh, la, poss, _, _ = R.predict(h, a, bool(gm.get("n")))
        rh, ra = E.expit(lh), E.expit(la)
        rt_h = float(E.ppp({c: rh[k] for k, c in enumerate(E.COMP)}) * poss); rt_a = float(E.ppp({c: ra[k] for k, c in enumerate(E.COMP)}) * poss)
        f = full.get(gm["id"]) or full.get(gid)
        if f and len(f) >= 7: bh, ba, ch, ca, poss, src = f[0], f[1], f[4], f[5], f[6], "stack"; miss = [f"{t}: {x.replace(';', ', ')}" for t, x in ((h, f[2]), (a, f[3])) if x]
        else: bh, ba, ch, ca, src, miss = rt_h, rt_a, list(rh), list(ra), "ratings", []
        cxh = comp_fix(FIX, dict(zip(COMP, ch)), poss); cxa = comp_fix(FIX, dict(zip(COMP, ca)), poss)
        hs, as_ = bh + cxh, ba + cxa
        H, A, p_ot = sim_game(hs, as_, poss, SDM, SDT, n=4000, seed=int(gid) % 100000)
        m, t = H - A, H + A; pq = lambda x, q: round(float(np.percentile(x, q)), 1)
        GA = dict(hs=round(hs, 2), p_home=round(float((m > 0).mean()), 4), p_ot=round(p_ot, 3), m10=pq(m, 10), m90=pq(m, 90), t10=pq(t, 10), t90=pq(t, 90),
                  h10=pq(H, 10), h90=pq(H, 90), a10=pq(A, 10), a90=pq(A, 90), src=src, base=[round(bh, 2), round(ba, 2)], fix=[round(cxh, 2), round(cxa, 2)],
                  ratings=[round(rt_h, 2), round(rt_a, 2)], poss=round(float(poss), 1))
        GA["as"] = round(as_, 2)
        tip_h = -hours_since(gm["k"]); dm = desk_margin(desk, h, a, bool(gm.get("n")))
        rest = lambda tm: (pd.Timestamp(gm["d"]) - last[tm]).days if tm in last else 3
        rec = {"h": h, "a": a, "d": gm["d"], "k": gm["k"], "A": GA, "desk_m": None if dm is None else round(dm, 2), "miss": [x.split(";") for x in (f[2], f[3])] if f and len(f) >= 4 else None,
               "gp": [gp.get(h, 0), gp.get(a, 0)], "rest": [rest(h), rest(a)], "B": None}
        ln = gm.get("ln")
        if ln and ln.get("bk"):
            mu_t, _ = market(ln["bk"], "t")
            if mu_t is not None: tot_gaps.append(hs + as_ - mu_t)
        recs[gid] = (gm, rec, miss, tip_h)
    # PRESEASON LEVEL GUARD (as CBB/NFL, but only when the whole slate sits 3+ pts off the market: on 2021-22 and 2026 openers the
    # plain gap beat a slate-median guard, so a normal slate keeps the raw gap)
    med = float(np.median(tot_gaps)) if len(tot_gaps) >= 6 else 0.0; TSHIFT = round(med, 2) if abs(med) >= 3 else 0.0
    out["total_guard"] = TSHIFT; out["slate_median_total_gap"] = round(med, 2)
    for gid, (gm, rec, miss, tip_h) in recs.items():
        ln = gm.get("ln"); GA = rec["A"]; hs, as_ = GA["hs"], GA["as"]
        if ln and ln.get("bk"):
            bk, po = ln["bk"], ln.get("po") or {}; B = {}
            mu_m, used = market(bk, "m")
            if mu_m is not None:
                ours = hs - as_; gap = ours - mu_m; side_home = gap > 0
                ls = (GA["base"][0] - GA["base"][1]) - (GA["ratings"][0] - GA["ratings"][1]) if GA["src"] == "stack" else 0.0
                dm = rec["desk_m"]
                feat = dict(engines_same_side=float(dm is not None and np.sign(dm - mu_m) == np.sign(gap)), lineup_drives=float(np.sign(ls) == np.sign(gap) and abs(ls) >= 1),
                            ls_abs=abs(ls), we_like_dog=float(np.sign(gap) != np.sign(mu_m) and abs(mu_m) > 0), gp_min=float(min(rec["gp"])),
                            gp_mid=float(16 <= min(rec["gp"]) <= 40), b2b_any=float(min(rec["rest"]) <= 1), fav_size=abs(mu_m), model_split=abs((dm if dm is not None else ours) - ours))
                share = conf_share(feat); bp = best_price(bk, po, side_home, "m", mu_m, tip_h); fresh = bool(bp and bp["fresh"])
                real = gap * (share if fresh else B_LATE)
                if np.sign(real) != np.sign(gap): real = 0.0     # never bet against our own number
                bp = best_price(bk, po, side_home, "m", mu_m + real, tip_h)
                early = feat["gp_min"] <= 5
                tier = ("A" if abs(real) >= Q90 and not early else "B" if abs(gap) >= 3 else "C") if fresh else "C"
                B["spread"] = dict(market=round(mu_m, 2), books=used, ours=round(ours, 2), gap=round(gap, 2), share=round(share if fresh else B_LATE, 3), real=round(real, 2),
                                   fresh=fresh, tier=tier, side=rec["h"] if side_home else rec["a"], bet=bp, agree=bool(feat["engines_same_side"]),
                                   reasons=reasons(feat, fresh, bp and bp["age_h"], tip_h, miss))
            mu_t, used_t = market(bk, "t")
            if mu_t is not None:
                ours_t = hs + as_; gap_t = ours_t - mu_t - TSHIFT; ov = gap_t > 0
                bp = best_price(bk, po, ov, "t", mu_t, tip_h); fresh = bool(bp and bp["fresh"])
                real = gap_t * (B_TOT_OPEN if fresh else 0.0)
                bp = best_price(bk, po, ov, "t", mu_t + real, tip_h)
                tier = ("A" if abs(gap_t) >= 4 else "B" if abs(gap_t) >= 3 else "C") if fresh else "C"
                Rt = [["+" if fresh else "-", (f"Line posted {bp['age_h']:.0f} h ago, tip in {tip_h:.0f} h: opener-like. Total gaps 4+ at the opener hit 56% (2021-22, 808) and 63% (2026 DraftKings, 310)" if fresh else
                                              "Close-like line: our NBA total gaps lost at the close at every size (2018-26)")] if bp and bp["age_h"] is not None else ["·", "No post time for this price"]]
                if TSHIFT: Rt.append(["·", f"Whole slate sits {TSHIFT:+.1f} vs the market's totals, so only the part beyond that counts (preseason level guard)"])
                if GA["src"] != "stack": Rt.append(["-", "No projected lineup for this game yet (ratings only)"])
                B["total"] = dict(market=round(mu_t, 2), books=used_t, ours=round(ours_t, 2), gap=round(gap_t, 2), share=B_TOT_OPEN if fresh else 0.0, real=round(real, 2), fresh=fresh,
                                  tier=tier, side="Over" if ov else "Under", bet=bp, reasons=Rt)
            sp_ = B.get("spread")
            if sp_:
                mu_B = sp_["market"] + sp_["real"]; best = None
                ph, pm = MLA.home_chance(bk, "NBA", lambda mu: ncdf(mu / SDM), mu_B, sp_["market"])   # anchored on the books' moneyline (J 2026-11-06)
                for b in MY_BOOKS:
                    r = bk.get(b)
                    if not r or not r[6] or not r[7]: continue
                    for home, pr in ((True, r[6]), (False, r[7])):
                        if home != (sp_["gap"] > 0): continue
                        p = ph if home else 1 - ph; ev = p * dec(pr) - 1
                        if best is None or ev > best["ev"]: best = dict(book=b, book_name=NAMES.get(b, b), pt=None, price=pr, p=round(p, 4), ev=round(ev, 4), age_h=None)
                if best:
                    tier = "B" if sp_["tier"] in "AB" and best["ev"] >= 0.03 and abs(mu_B) <= 7 else "C"
                    B["ml"] = dict(mkt_p=None if pm is None else round(pm if sp_["gap"] > 0 else 1 - pm, 4), market=sp_["market"], ours=sp_["ours"], gap=sp_["gap"], real=sp_["real"], fresh=sp_["fresh"], tier=tier, side=sp_["side"], bet=best,
                                   reasons=[["·", "Moneyline prices the spread's trusted edge as a win chance; not tested on NBA openers, so a lean at most and only near pick'em. Win chance = the books' no-vig moneyline moved only by that trusted edge"]])
            ts = [z for z in B.values() if z["tier"] != "C" and z["bet"]]
            B["best"] = min(ts, key=lambda z: ("AB".index(z["tier"]), -z["bet"]["ev"]))["tier"] if ts else "C"
            rec["B"] = B
        out["games"][gid] = rec
    clean = lambda o: {k: clean(x) for k, x in o.items()} if isinstance(o, dict) else [clean(x) for x in o] if isinstance(o, list) else (None if isinstance(o, float) and o != o else o)
    import unproven; unproven.annotate("NBA", out)   # unproven rating on tier C passes (J 2026-11-06)
    # J 2026-11-06: a started game keeps its last pre-game record (frozen) so the desk can list the whole day (models/model-b/pregame.py)
    from pregame import keep
    json.dump(clean(dict(out, games=keep(f"{APP}/nba_ab.json", out["games"], kick=lambda i, r: r.get("k")))), open(f"{APP}/nba_ab.json", "w"), separators=(",", ":"), allow_nan=False)
    os.makedirs(f"{HERE}/history", exist_ok=True)
    with open(f"{HERE}/history/runs.jsonl", "a") as fh:
        for gid, g in out["games"].items():
            fh.write(json.dumps({"t": out["made"], "id": gid, "h": g["h"], "a": g["a"], "d": g["d"], "A": [g["A"]["hs"], g["A"]["as"]], "base": g["A"]["base"], "desk_m": g["desk_m"],
                                 "B": {k: {kk: z.get(kk) for kk in ("market", "gap", "real", "fresh", "tier", "side", "bet")} for k, z in (g["B"] or {}).items() if isinstance(z, dict)}}) + "\n")
    tiers = {}
    for g in out["games"].values():
        for k in ("spread", "total", "ml"):
            z = g["B"] and g["B"].get(k)
            if z: tiers[f"{k}_{z['tier']}"] = tiers.get(f"{k}_{z['tier']}", 0) + 1
    print(len(out["games"]), "games ->", f"{APP}/nba_ab.json | lineup feed", bool(full), "| data through", out["data_through"], "| tiers", tiers, "| slate total gap median", out["slate_median_total_gap"], "guard", TSHIFT)
    for gid, g in out["games"].items():
        z = g["B"] and g["B"].get("spread"); t_ = g["B"] and g["B"].get("total")
        print(f"  {g['d']} {g['a']}@{g['h']} A {g['A']['as']:.1f}-{g['A']['hs']:.1f} ({g['A']['src']}, fix {g['A']['fix']}) desk {g['desk_m']}" +
              (f" | mkt {z['market']:+.1f} gap {z['gap']:+.1f} real {z['real']:+.1f} {z['tier']} {z['side']} fresh={z['fresh']}" if z else "") +
              (f" | tot mkt {t_['market']:.1f} gap {t_['gap']:+.1f} {t_['tier']} {t_['side']}" if t_ else ""))

if __name__ == "__main__": main()
