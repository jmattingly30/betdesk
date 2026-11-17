"""Shot-state NHL game simulator (5-second ticks), vectorized over games x sims.
State: clock, period, score, active minors per team (up to 2, 120 s, PP goal ends one), pulled goalie, 3v3 OT, shootout.
Per tick each team: unblocked-attempt hazard (state base x team volume ratings) x shot quality (state base x team quality)
x finishing x opposing starting goalie -> goal hazard; penalties by score state x taken/drawn ratings.
Usage (module): run(games_df, ratings_df, base, params, N) -> per-game prediction frame"""
import numpy as np, pandas as pd
DT = 5
PULL1, PULL2 = 120, 175     # seconds left when trailing team pulls (down 1 / down 2)
EN_G60 = 22.4               # goals/60 vs empty net (pulled table)
PULL_G60 = 8.6              # xG/60 for the 6v5 team (scaled by off mult)
OT_G60 = 6.2                # 3v3 goals/60 per team
PEN60 = 3.05                # minors per team per 60 at tied 5v5 (tuned so PP time matches)
PEN_SD = {-3: 1.30, -2: 1.22, -1: 1.18, 0: 1.0, 1: 0.85, 2: 0.82, 3: 0.80}  # by drawer perspective? -> applied to TAKER's score diff below (inverted)

def run(G, base, P, N=400, seed=0):
    """G: frame with per-game home/away multipliers: off5_h, def5_h (allowed, multiplies opp), pp_h, pk_h, fin_h, gk_h, pt_h, pd_h,
    rest_h (scoring multiplier for rest, 1 = none) ... same _a; season. Set P["dist"] to also get the full team-score distributions."""
    rng = np.random.default_rng(seed)
    n = len(G); M = n * N
    rep = lambda c: np.repeat(G[c].to_numpy(float), N)
    season = np.repeat(G.season.to_numpy(), N)
    sd_list = np.arange(-3, 4)
    # base tables -> arrays [season_idx, sd+3]
    seasons = sorted(set(G.season))
    si = np.searchsorted(seasons, season)
    bfq = np.array([[base["b_fen"].get(f"{int(s)}|{d}", np.nan) * base["b_q"].get(f"{int(s)}|{d}", np.nan) for d in sd_list] for s in seasons])  # 5v5 xg per second
    man_x = lambda s, m: base["b_xgm"].get(f"{int(s)}|{m}", np.nan)
    bpp = np.array([[man_x(s, "5v4"), man_x(s, "5v3"), man_x(s, "4v5"), man_x(s, "3v5"), man_x(s, "4v4"), man_x(s, "3v3"), man_x(s, "4v3"), man_x(s, "3v4")] for s in seasons])
    # per-sim rate noise (game-level uncertainty in team strength)
    noise = lambda: np.exp(rng.normal(0, P["rate_sd"], M) - P["rate_sd"] ** 2 / 2)
    k = P["scale"]
    off_h = rep("off5_h") * rep("def5_a") * noise() * k; off_a = rep("off5_a") * rep("def5_h") * noise() * k
    pp_h = rep("pp_h") * rep("pk_a") * k; pp_a = rep("pp_a") * rep("pk_h") * k
    fin_h = rep("fin_h") * rep("gk_a"); fin_a = rep("fin_a") * rep("gk_h")
    pen_h = rep("pt_h") * rep("pd_a"); pen_a = rep("pt_a") * rep("pd_h")
    hfa = P["hfa"]
    off_h *= hfa; off_a /= hfa
    if "rest_h" in G:      # rest lowers a tired team's whole attack: even strength, power play, overtime
        rh, ra = rep("rest_h"), rep("rest_a"); off_h *= rh; off_a *= ra; pp_h *= rh; pp_a *= ra
    sh = np.zeros(M, int); sa = np.zeros(M, int)
    ph = np.zeros((M, 2)); pa = np.zeros((M, 2))   # remaining penalty seconds (two slots)
    sog_h = np.zeros(M); sog_a = np.zeros(M)
    pen_scale = PEN60 / 3600 * DT * P["pen_mult"]
    psd = np.array([PEN_SD[d] for d in sd_list])
    for tick in range(3600 // DT):
        tsec = tick * DT; left = 3600 - tsec
        nh = (ph > 0).sum(1); na = (pa > 0).sum(1)
        skh = 5 - nh; ska = 5 - na
        d = np.clip(sh - sa, -3, 3)
        p3 = tsec >= 2400
        pull_h = p3 & (((d == -1) & (left <= PULL1)) | ((d == -2) & (left <= PULL2))) & (nh == 0)
        pull_a = p3 & (((d == 1) & (left <= PULL1)) | ((d == 2) & (left <= PULL2))) & (na == 0)
        # goal hazards (per tick)
        ev = (nh == na)
        b5h = bfq[si, d + 3]; b5a = bfq[si, -d + 3]
        lam_h = np.where(ev, b5h * (1.0 if True else 1) * np.where(skh == 4, bpp[si, 4] / bfq[si, 3], 1.0), 0.0) * off_h
        lam_a = np.where(ev, b5a * np.where(ska == 4, bpp[si, 4] / bfq[si, 3], 1.0), 0.0) * off_a
        # special teams
        adv_h = skh > ska; adv_a = ska > skh
        pp_rate_h = np.select([(skh == 5) & (ska == 4), (skh == 5) & (ska == 3), (skh == 4) & (ska == 3)], [bpp[si, 0], bpp[si, 1], bpp[si, 6]], 0.0)
        pp_rate_a = np.select([(ska == 5) & (skh == 4), (ska == 5) & (skh == 3), (ska == 4) & (skh == 3)], [bpp[si, 0], bpp[si, 1], bpp[si, 6]], 0.0)
        sh_rate_h = np.select([(skh == 4) & (ska == 5), (skh == 3) & (ska == 5), (skh == 3) & (ska == 4)], [bpp[si, 2], bpp[si, 3], bpp[si, 7]], 0.0)
        sh_rate_a = np.select([(ska == 4) & (skh == 5), (ska == 3) & (skh == 5), (ska == 3) & (skh == 4)], [bpp[si, 2], bpp[si, 3], bpp[si, 7]], 0.0)
        lam_h = lam_h + adv_h * pp_rate_h * pp_h + adv_a * sh_rate_h * k
        lam_a = lam_a + adv_a * pp_rate_a * pp_a + adv_h * sh_rate_a * k
        # pulled goalie
        lam_h = np.where(pull_h, PULL_G60 / 3600 * off_h / k * k, lam_h); lam_a = np.where(pull_a, PULL_G60 / 3600 * off_a, lam_a)
        lam_h = np.where(pull_a, EN_G60 / 3600, lam_h * fin_h); lam_a = np.where(pull_h, EN_G60 / 3600, lam_a * fin_a)
        lam_h = np.where(pull_h & pull_a, 4 / 3600, lam_h); lam_a = np.where(pull_h & pull_a, 4 / 3600, lam_a)
        gh = rng.random(M) < lam_h * DT; ga = rng.random(M) < lam_a * DT
        both = gh & ga; ga &= ~(both & (rng.random(M) < 0.5)); gh &= ~(both & ga)
        sh += gh; sa += ga
        # power-play goal ends one opposing minor
        e = gh & adv_h; i = np.argmax(pa > 0, 1); pa[np.where(e)[0], i[e]] = 0
        e = ga & adv_a; i = np.argmax(ph > 0, 1); ph[np.where(e)[0], i[e]] = 0
        ph = np.maximum(ph - DT, 0); pa = np.maximum(pa - DT, 0)
        # penalties (taker's score diff: leading teams take more? -> use drawer's state table inverted)
        dh = np.clip(sh - sa, -3, 3)
        nh = (ph > 0).sum(1); na = (pa > 0).sum(1)
        qh = (rng.random(M) < pen_scale * pen_h * psd[-dh + 3]) & (nh < 2) & (left > 20)
        qa = (rng.random(M) < pen_scale * pen_a * psd[dh + 3]) & (na < 2) & (left > 20)
        i = np.argmin(ph, 1); ph[np.where(qh)[0], i[qh]] = 120
        i = np.argmin(pa, 1); pa[np.where(qa)[0], i[qa]] = 120
    reg_h, reg_a = sh.copy(), sa.copy()
    tie = sh == sa
    # 3v3 OT, 300 s, first goal wins
    o_h = OT_G60 / 3600 * off_h / k * fin_h * P["ot_mult"]; o_a = OT_G60 / 3600 * off_a / k * fin_a * P["ot_mult"]
    tot = o_h + o_a
    p_goal = 1 - np.exp(-tot * 300)
    otg = tie & (rng.random(M) < p_goal)
    hw = rng.random(M) < o_h / tot
    sh += otg & hw; sa += otg & ~hw
    so = tie & ~otg
    sow = rng.random(M) < 0.5
    sh += so & sow; sa += so & ~sow
    out = pd.DataFrame({"game_id": G.game_id.values})
    R = lambda x: x.reshape(n, N)
    H, A = R(sh), R(sa)
    out["pred_h"] = H.mean(1); out["pred_a"] = A.mean(1)
    out["p_home"] = (H > A).mean(1); out["p_ot"] = R(tie).mean(1)
    out["med_tot"] = np.median(H + A, 1)
    out["sd_tot"] = (H + A).std(1)
    # total distribution for line grading
    tot_ = H + A
    for L in (4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5):
        out[f"over_{L}"] = (tot_ > L).mean(1); out[f"push_{L}"] = (tot_ == L).mean(1)
    out["pred_reg_h"] = R(reg_h).mean(1); out["pred_reg_a"] = R(reg_a).mean(1)
    if P.get("dist"):      # chance of 0, 1, ... 9, 10+ goals for each team (overtime and shootout goal included)
        out["dist_h"] = [np.bincount(np.minimum(x, 10), minlength=11) / N for x in H]
        out["dist_a"] = [np.bincount(np.minimum(x, 10), minlength=11) / N for x in A]
        out["dist_t"] = [np.bincount(np.minimum(x, 15), minlength=16) / N for x in tot_]   # game total 0 ... 14, 15+ (joint, not two marginals)
        out["p_h2"] = (H - A >= 2).mean(1); out["p_a2"] = (A - H >= 2).mean(1)            # puck line: wins by 2+
    return out
