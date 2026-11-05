"""CBB Model A + Model B for Bet Desk (J 2026-11-02: CBB is the first sport on the two-model system).
Model A = projected scores: CBB v2 means (app/data/cbb_v2.json from models/rebuild/basketball/live_cbb.py) + the possession
          sim around them (end-game fouling + overtime, spread calibrated to v2's measured miss) for win %, ranges, OT chance.
Model B = bets built on Model A: sharp-weighted devigged market number (devig.WEIGHT, capped 1.5x), the share of our gap that
          is real (confidence fit on the 2026 DK opener, model_b_test.json 'live'), line age (fresh <= 24 h behaves like an opener,
          older behaves like a close), then EV at J's books and a tier: A = bet, B = lean (tracked, not a bet), C = pass.
Run after app/export_slates.py (needs game_slates.json ln.bk / ln.po):  python3 models/cbb_ab/live.py  -> app/data/cbb_ab.json"""
import json, os, sys, math, datetime as dt, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, f"{HERE}/../model-b")
from lib import sim_game, ncdf
import devig as DV
ROOT = os.path.normpath(f"{HERE}/../..")
APP = f"{ROOT}/app/data"
# J 2026-11-02: every Colorado-legal book counts (no 'my books' subset)
MY_BOOKS = ['draftkings', 'fanduel', 'betmgm', 'williamhill_us', 'espnbet', 'fanatics', 'betrivers', 'ballybet', 'bet365', 'hardrockbet', 'circasports']
NAMES = dict(pinnacle='Pinnacle', betonlineag='BetOnline', draftkings='DraftKings', fanduel='FanDuel', betmgm='BetMGM', williamhill_us='Caesars',
             espnbet='ESPN BET', fanatics='Fanatics', betrivers='BetRivers', ballybet='Bally Bet', hardrockbet='Hard Rock', bet365='bet365', circasports='Circa')
T_A = json.load(open(f"{HERE}/model_a_test.json")); T_B = json.load(open(f"{HERE}/model_b_test.json"))
SD_M, SD_T = T_A["sd_margin_fit"], T_A["sd_total_fit"]
LV = T_B["live"]; Q90 = LV["q90"]
FRESH_H = 24            # a price posted within 24 h is treated as an opener (Look ahead uses the same window)
B_CLOSE, B_TOT_OPEN = 0.13, 0.505   # share of our gap that is real: spread at the close (research), total at the opener
NOW = dt.datetime.now(dt.timezone.utc)
tier_csv = pd.read_csv(f"{ROOT}/models/cbb/data/cbb_team_tier.csv")
TIER = {r.name.lower(): ({"high": 2, "mid": 1, "low": 0}[r.tier], r.conf_id) for r in tier_csv.itertuples()}
inv = lambda p: math.sqrt(2) * _erfinv(2 * p - 1)
def _erfinv(y):
    a = 0.147; s = math.copysign(1, y); l = math.log(1 - y * y); t = 2 / (math.pi * a) + l / 2
    return s * math.sqrt(math.sqrt(t * t - l / a) - t)
dec = DV.dec_from_am
# miss grows with the spread (2013-26 lined games: sd of actual - close 10.9 up to 12-way favs, 11.9 at 18-25, 12.9 at 25+)
sd_m_for = lambda L: float(np.interp(abs(L), [0, 12, 15, 21.5, 30], [SD_M, SD_M, 11.1, 11.9, 12.9]))

def hours_since(ts):
    if not ts: return None
    try: return (NOW - dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 3600
    except ValueError: return None

def market(bk, kind):
    """sharp-weighted devigged market number from every book: home margin (kind 'm') or total ('t').
    Each book's line + its two prices -> the expected number it implies (power devig, normal spread), averaged with devig.WEIGHT."""
    acc = tw = 0.0; used = []
    for b, r in bk.items():
        pt, p1, p2 = (r[0], r[1], r[2]) if kind == "m" else (r[3], r[4], r[5])
        if pt is None or not p1 or not p2: continue
        p = DV.power([dec(p1), dec(p2)])[0]          # home covers / over
        p = min(max(p, 0.02), 0.98)
        mu = (-pt + sd_m_for(pt) * inv(p)) if kind == "m" else (pt + SD_T * inv(p))
        w = min(DV.WEIGHT.get(b, 0.8), DV.MAX_RATIO); acc += w * mu; tw += w; used.append(b)
    return (acc / tw, used) if tw else (None, [])

def conf_share(feat):
    """b0 + sum b_k z_k: how much of our gap the 2026 opener says is real for this game type."""
    c = LV["coef"]; s = c["edge"]
    for f in LV["features"]: s += c[f] * (feat[f] - LV["mu"][f]) / LV["sd"][f]
    return s

def reasons(feat, fresh, age):
    R = []
    if feat["we_like_dog"]: R.append(["+", "We like the underdog: in basketball our underdog gaps have been far more real than favorite gaps"])
    else: R.append(["-", "We like the favorite: favorite gaps have held up less than underdog gaps"])
    if feat["neu"]: R.append(["+", "Neutral site: our biggest-hitting game type at the opener"])
    if feat["engines_same_side"]: R.append(["+", "Both our engines (previous CBB model and the rebuild) are on this side of the line"])
    else: R.append(["-", "Our two engines split on this side of the line"])
    tm = feat["tier_min"]
    R.append(["+" if tm >= 2 else "-" if tm == 0 else "·", {2: "Both teams high-major (gaps most real there)", 1: "Lowest team is mid-major", 0: "A low-major team is involved (gaps least real there)"}[int(tm)]])
    if feat["nonconf"]: R.append(["+", "Non-conference game"])
    if feat["nov"]: R.append(["·", "November: ratings still lean on preseason roster priors"])
    if age is None: R.append(["·", "No post time for this price"])
    elif fresh: R.append(["+", f"Price posted {age:.0f} h ago: still opener-like, where our edge has been real"])
    else: R.append(["-", f"Price has been up {age:.0f} h: close-like, where most of our gap has not held"])
    R.append(["·", "No live CBB injury feed yet: players out are not in either model"])
    return R

def best_price(bk, po, side_home, kind, mu_B):
    """best EV at J's books for one side, with that book's line age"""
    out = None
    for b in MY_BOOKS:
        r = bk.get(b); 
        if not r: continue
        pt, ph, pa = (r[0], r[1], r[2]) if kind == "m" else (r[3], r[4], r[5])
        if pt is None or not ph or not pa: continue
        if kind == "m":
            p_home = ncdf((mu_B + pt) / sd_m_for(mu_B)); p, pr, lab = (p_home, ph, pt) if side_home else (1 - p_home, pa, -pt)
        else:
            p_ov = 1 - ncdf((pt - mu_B) / SD_T); p, pr, lab = (p_ov, ph, pt) if side_home else (1 - p_ov, pa, pt)
        ev = p * dec(pr) - 1; age = hours_since((po.get(b) or [None] * 3)[0 if kind == "m" else 1])
        if out is None or ev > out["ev"]: out = dict(book=b, book_name=NAMES.get(b, b), pt=lab, price=pr, p=round(p, 4), ev=round(ev, 4), age_h=None if age is None else round(age, 1))
    return out

def level_guard(V, slate, min_n=12):
    mg, mm, tg = [], [], []
    for gid, v in V["games"].items():
        x = slate.get(gid); ln = x and x.get("ln")
        if not (ln and ln.get("bk")): continue
        m, _ = market(ln["bk"], "m"); t, _ = market(ln["bk"], "t")
        if m is not None: mg.append(v["hs"] - v["as"] - m); mm.append(m)
        if t is not None: tg.append(v["hs"] + v["as"] - t)
    G = {"n_spread": len(mg), "n_total": len(tg), "total_shift": 0.0, "spread_a": 0.0, "spread_b": 0.0}
    if len(tg) >= min_n: G["total_shift"] = round(float(np.median(tg)), 2)
    if len(mg) >= min_n:
        b, a = np.polyfit(mm, mg, 1)          # live gap = a + b * market margin
        G["spread_b"] = round(float(b - (-0.046)), 3); G["spread_a"] = round(float(a - 0.514), 2)   # excess over history
    return G

def main():
    V = json.load(open(f"{APP}/cbb_v2.json")); S = json.load(open(f"{APP}/game_slates.json"))["CBB"]["games"]
    slate = {x["id"]: x for x in S}
    out = {"made": NOW.strftime("%Y-%m-%d %H:%M UTC"), "from": {"cbb_v2": V.get("made"), "slates": S and "game_slates.json"},
           "how": {"A": "CBB v2 scores + possession sim (end-game fouling, overtime); spread calibrated to v2's miss "
                        f"(margin sd {SD_M}, total sd {SD_T})",
                   "B": "Sharp-weighted devigged market number; real share of our gap from the 2026 opener confidence fit; "
                        f"fresh price <= {FRESH_H} h acts like an opener. Tier A = bet, B = lean (tracked, not a bet), C = pass."},
           "tested": {"A_team_MAE": T_A["test_2020_2026"]["v2"]["team_MAE_lined"], "line_team_MAE": T_A["test_2020_2026"]["line"]["team_MAE_lined"],
                      "A_team_APE": T_A["test_2020_2026"]["v2"]["team_APE_lined"], "line_team_APE": T_A["test_2020_2026"]["line"]["team_APE_lined"],
                      "B_spread_tiers_open_2026": T_B["R4_A_conf_top10_or_gap4__B_gap3"]},
           "games": {}}
    nomatch = 0
    # PRESEASON LEVEL GUARD (J 2026-11-02 04:15 UTC check): live v2 sat 6.2 pts under every CBB total and 5.9 under big home
    # favorites, while 2018-26 history averages ~0 to +1 against the line (v2 = 0.954 x line + 0.51). That is a level / squeeze
    # problem in this season's preseason ratings, not 40 separate edges, so Model B measures each gap against the slate-wide
    # pattern: totals minus the slate's median gap; spreads minus the slate's excess squeeze vs history. Model A is untouched.
    LG = level_guard(V, slate)
    out["level_guard"] = LG
    for gid, v in V["games"].items():
        hs, as_, poss = v["hs"], v["as"], v.get("poss") or 68
        H, A = sim_game(hs, as_, poss, SD_M, SD_T, n=4000, seed=int(gid) % 100000)
        m, t = H - A, H + A
        pq = lambda x, q: round(float(np.percentile(x, q)), 1)
        GA = dict(hs=hs, as_=as_, p_home=round(float((m > 0).mean()), 4), p_ot=round(float(ncdf(0.5 / SD_M) - ncdf(-0.5 / SD_M)) * 1.25, 3),
                  m10=pq(m, 10), m90=pq(m, 90), t10=pq(t, 10), t90=pq(t, 90), h10=pq(H, 10), h90=pq(H, 90), a10=pq(A, 10), a90=pq(A, 90),
                  today=[v.get("today_hs"), v.get("today_as")], rb=[v.get("rb_hs"), v.get("rb_as")], poss=poss)
        GA["as"] = GA.pop("as_")
        rec = {"h": v["h"], "a": v["a"], "d": v["d"], "n": v.get("n", 0), "A": GA, "B": None}
        x = slate.get(gid); ln = x and x.get("ln")
        if ln and ln.get("bk"):
            bk, po = ln["bk"], ln.get("po") or {}
            th, ta = TIER.get(v["h"].lower()), TIER.get(v["a"].lower())
            if th is None or ta is None: nomatch += 1
            B = {}
            mu_m, used_m = market(bk, "m")
            if mu_m is not None:
                ours = hs - as_; gap = ours - mu_m - LG["spread_a"] - LG["spread_b"] * mu_m; side_home = gap > 0
                tm_ = (v["today_hs"] - v["today_as"]) - mu_m - LG["spread_a"] - LG["spread_b"] * mu_m if v.get("today_hs") is not None else gap
                rm_ = (v["rb_hs"] - v["rb_as"]) - mu_m - LG["spread_a"] - LG["spread_b"] * mu_m if v.get("rb_hs") is not None else gap
                feat = dict(we_like_dog=float(np.sign(gap) != np.sign(mu_m) and abs(mu_m) > 0), neu=float(bool(v.get("n"))),
                            nonconf=float(th is not None and ta is not None and th[1] != ta[1]), tier_min=float(min(th[0] if th else 1, ta[0] if ta else 1)),
                            engines_same_side=float(np.sign(tm_) == np.sign(rm_)), av_abs=0.0, nov=float(v["d"][5:7] == "11"), fav_size=abs(mu_m))
                share = conf_share(feat); bp = best_price(bk, po, side_home, "m", mu_m)
                fresh = bp is not None and bp["age_h"] is not None and bp["age_h"] <= FRESH_H
                real = gap * share if fresh else gap * B_CLOSE
                if np.sign(real) != np.sign(gap): real = 0.0          # never bet against our own number (flipped sides lost: 43%)
                bp = best_price(bk, po, side_home, "m", mu_m + real)
                if fresh: tier = "A" if (abs(real) >= Q90 or abs(gap) >= 4) else "B" if abs(gap) >= 3 else "C"
                else: tier = "A" if abs(gap) >= 6 else "B" if abs(gap) >= 4 else "C"
                B["spread"] = dict(market=round(mu_m, 2), books=used_m, ours=round(ours, 2), gap=round(gap, 2), share=round(share if fresh else B_CLOSE, 3),
                                   real=round(real, 2), fresh=fresh, tier=tier, side=v["h"] if side_home else v["a"], bet=bp,
                                   reasons=reasons(feat, fresh, bp and bp["age_h"]))
            mu_t, used_t = market(bk, "t")
            if mu_t is not None:
                ours_t = hs + as_; gap_t = ours_t - mu_t - LG["total_shift"]; over = gap_t > 0
                bp = best_price(bk, po, over, "t", mu_t)
                fresh = bp is not None and bp["age_h"] is not None and bp["age_h"] <= FRESH_H
                real = gap_t * (B_TOT_OPEN if fresh else 0.0)   # totals at the close: no real share (all gap sizes lost 2020-26)
                bp = best_price(bk, po, over, "t", mu_t + real)
                tier = ("A" if abs(gap_t) >= 6 else "B" if abs(gap_t) >= 3 else "C") if fresh else "C"
                B["total"] = dict(market=round(mu_t, 2), books=used_t, ours=round(ours_t, 2), gap=round(gap_t, 2), real=round(real, 2), fresh=fresh, tier=tier,
                                  side="Over" if over else "Under", bet=bp,
                                  reasons=[["+" if fresh else "-", (f"Price posted {bp['age_h']:.0f} h ago: opener-like (totals gaps 3+ hit 55% at the 2026 opener)" if fresh else
                                            "Price is close-like: our total gaps have lost at the close at every size")] if bp and bp["age_h"] is not None else ["·", "No post time for this price"]])
            sp_ = B.get("spread")
            if sp_:
                mu_B = sp_["market"] + sp_["real"]; sd = sd_m_for(mu_B); ph = 1 - ncdf((0 - mu_B) / sd)
                best = None
                for b in MY_BOOKS:
                    r = bk.get(b)
                    if not r or not r[6] or not r[7]: continue
                    for home, pr in ((True, r[6]), (False, r[7])):
                        p = ph if home else 1 - ph; ev = p * dec(pr) - 1
                        if (home == (sp_["gap"] > 0)) and (best is None or ev > best["ev"]):
                            best = dict(book=b, book_name=NAMES.get(b, b), pt=None, price=pr, p=round(p, 4), ev=round(ev, 4), age_h=None)
                if best:
                    # ML is the spread's edge priced as a win; never tested on CBB openers -> lean at most, and only near pick'em
                    tier = "B" if sp_["tier"] in "AB" and best["ev"] >= 0.03 and abs(mu_B) <= 7 else "C"
                    B["ml"] = dict(market=sp_["market"], ours=sp_["ours"], gap=sp_["gap"], real=sp_["real"], fresh=sp_["fresh"], tier=tier, side=sp_["side"], bet=best,
                                   reasons=[["·", "Moneyline uses the spread's trusted edge as a win chance; not yet tested on CBB openers, so it is a lean at most"]])
            ts = [z for z in B.values() if z["tier"] != "C" and z["bet"]]
            B["best"] = min(ts, key=lambda z: ("AB".index(z["tier"]), -z["bet"]["ev"]))["tier"] if ts else "C"
            rec["B"] = B
        out["games"][gid] = rec
    json.dump(out, open(f"{APP}/cbb_ab.json", "w"), separators=(",", ":"))
    # tracking (J model-tracking rule): every run's Model A scores + Model B picks, first and latest seen, graded by grade.py
    os.makedirs(f"{HERE}/history", exist_ok=True)
    with open(f"{HERE}/history/runs.jsonl", "a") as fh:
        for gid, g in out["games"].items():
            fh.write(json.dumps({"t": out["made"], "id": gid, "h": g["h"], "a": g["a"], "d": g["d"], "A": [g["A"]["hs"], g["A"]["as"]],
                                 "B": {k: {kk: z[kk] for kk in ("market", "gap", "real", "fresh", "tier", "side", "bet")} for k, z in (g["B"] or {}).items() if isinstance(z, dict)}}) + "\n")
    n = len(out["games"]); nb = sum(1 for g in out["games"].values() if g["B"])
    tiers = {}
    for g in out["games"].values():
        for k in ("spread", "total", "ml"):
            z = g["B"] and g["B"].get(k)
            if z: tiers[f"{k}_{z['tier']}"] = tiers.get(f"{k}_{z['tier']}", 0) + 1
    print(n, "games ->", f"{APP}/cbb_ab.json |", nb, "with lines | tiers", tiers, "| teams without tier match", nomatch, "| level guard", LG)
    for gid, g in list(out["games"].items())[:40]:
        z = g["B"] and g["B"].get("spread")
        if z and z["tier"] != "C": print(" ", g["a"], "@", g["h"], z["tier"], z["side"], "gap", z["gap"], "real", z["real"], z["bet"])

if __name__ == "__main__": main()
