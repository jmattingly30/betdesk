"""CFB Model A shared pieces (J 2026-11-05 rollout; fourth sport on the two-model system after CBB, NFL, NBA).
- v4 means: the tested CFB rebuild terms on top of the desk's v3 (unblended): program SRS (s1, s23), coach career win%
  (c_cwin), crowd size (x_att); weeks 1-3 also keep v3's roster terms (RT + D). Total = v3 total + in-season tempo (x_ip).
  Same as models/rebuild/football/cfb_gap/live/cfb_live.py (which writes app/data/cfb_live.json for live games).
- drive simulator (research/cfb-sim-2026-10/drive_sim.py, J yardstick regrade 2026-10-31): league drive tables 2023-25,
  TD/FG odds tilted so each team's expected points match Model A, shared game shock 3.9 + team shock 2.0, bias .56, margin
  slope (SHRINK) fitted on 2014-21. Used for ranges, win %, team-total over chances, key numbers. Means stay Model A's.
  GAME STATE (J 2026-11-15, football sim roadmap step 2): outcomes, drive lengths and the try after a TD depend on the game
  state (normal; 1st-half last 2:00; last 5:00 trailing 1-3 / 4-8 / 9+, leading, tied) incl. an end-of-half outcome (kneel /
  clock out), from the same 2023-25 drives; the 2nd-half kickoff goes to the team that kicked first (was always the away
  team); after a return TD the scorer kicks off. Bias re-set to -0.10 (sim team mean +0.2 over Model A, as before).
  OVERTIME (step 3): real college overtime (see Sim.overtime), so no simulated game ends tied.
  Test 2022-25 (600 sims/game): exact final log score -6.947 -> -6.892, team score -3.550 -> -3.537, margin -4.164 -> -4.125,
  team-total over Brier .2570 -> .2561; margin of 3 4.0% -> 6.0% (real 9.8%), of 7 4.5% -> 5.7% (real 7.9%); field goal
  when down 4-8 in the last 5:00 real 0.2%, sim 0.3%; 1st-half last 2:00 field goals 3% -> 9% (real 12%).
- frames: data/te_feat.parquet (2022-25 lined games) and data/fit_feat.parquet (2014-21), dumped from the cfb_gap prelude
  (python3 prep.py)."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = ['s1', 's23']; RT = ['d_ro', 'd_rp']
D_ = ['d_' + c for c in ['qb_anya', 'qb_lvl', 'qb_new', 'qb_none', 'avail_skill', 'xfer_skill']]
E4 = ['model_margin'] + P + RT + D_ + ['c_cwin', 'x_att']      # weeks 1-3
L4 = ['model_margin'] + P + ['c_cwin', 'x_att']                 # weeks 4+ (live)
TT = ['model_total', 'x_ip']

def ridge(F, cols, y, lam=1e-4):
    X = np.c_[np.ones(len(F)), F[cols].values]; return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ F[y].values)
def apply(T, cols, b): return np.c_[np.ones(len(T)), T[cols].values] @ b
def v4(fit, te):
    """Model A means on te from a fit on fit (split weeks 1-3 / 4+ as tested): margin, total"""
    m = np.zeros(len(te)); t = np.zeros(len(te))
    for fm, tm, cols in ((fit.wk <= 3, te.wk <= 3, E4), (fit.wk > 3, te.wk > 3, L4)):
        m[tm.values] = apply(te[tm], cols, ridge(fit[fm], cols, 'am'))
        t[tm.values] = apply(te[tm], TT, ridge(fit[fm], TT, 'at'))
    return m, t

# ---------------- drive simulator
SIGC, SIGI, BIAS = 3.9, 2.0, -0.10
CAT = {'TD': 0, 'FG': 1, 'PUNT': 2, 'PUNT TD': 2, 'PUNT RETURN TD': 2, 'DOWNS': 3, 'DOWNS TD': 3, 'MISSED FG': 3, 'MISSED FG TD': 3,
       'INT': 4, 'FUMBLE': 4, 'INT TD': 4, 'FUMBLE RETURN TD': 4, 'FUMBLE TD': 4, 'SF': 5, 'END OF HALF': 6, 'END OF GAME': 6, 'END OF 4TH QUARTER': 6}
NB = 20; PAT = np.array([6, 7, 8]); PATP = np.array([0.04, 0.92, 0.04]); NST = 7
STATES = ["normal", "1st half last 2:00", "last 5:00 trailing 1-3", "last 5:00 trailing 4-8", "last 5:00 trailing 9+", "last 5:00 leading", "last 5:00 tied"]

def state(h2, t, lead):
    """h2: 2nd half (bool array), t: seconds left in the half, lead: offense minus defense score -> state 0-6"""
    s = np.zeros(len(t), int); late = h2 & (t <= 300)
    s[(~h2) & (t <= 120)] = 1
    s[late & (lead <= -1) & (lead >= -3)] = 2; s[late & (lead <= -4) & (lead >= -8)] = 3; s[late & (lead <= -9)] = 4
    s[late & (lead >= 1)] = 5; s[late & (lead == 0)] = 6
    return s

class Sim:
    def __init__(self, drives=f"{HERE}/data/drives_2023_25.parquet", seed=11):
        self.rng = np.random.default_rng(seed)
        D = pd.read_parquet(drives); D = D[D.season.between(2023, 2025) & (D.period <= 4) & D.y0.between(1, 99)].copy().sort_values(['game_id', 'n'])
        D['c'] = D.res.map(CAT); D['half'] = np.where(D.period <= 2, 1, 2)
        D['next_y'] = D.groupby(['game_id', 'half']).y0.shift(-1); D['sec'] = (D.t0 - D.t1).clip(5, 600)
        K = D[D.c.notna()].copy(); K['c'] = K.c.astype(int); K['b'] = (K.y0 // 5).clip(0, NB - 1).astype(int)
        K['s'] = state(K.half.values == 2, np.where(K.half == 1, K.t0 - 1800, K.t0), K.sd0.values); K['bb'] = np.digitize(K.y0, [40, 60, 75])
        base = np.zeros((NB, 7))
        for b in range(NB):
            v = K[(K.b == b) & (K.s == 0)].c.value_counts().reindex(range(7), fill_value=0).values + 0.5; base[b] = v / v.sum()
        self.PO = np.zeros((NST, NB, 7)); self.PO[0] = base
        for s in range(1, NST):   # state = normal shape x the state's outcome mix by field zone (shrunk toward normal)
            for bb in range(4):
                y0 = K[(K.s == 0) & (K.bb == bb)].c.value_counts(normalize=True).reindex(range(7), fill_value=0).values
                cs = K[(K.s == s) & (K.bb == bb)].c.value_counts().reindex(range(7), fill_value=0).values + 2.0 * y0
                mult = (cs / cs.sum()) / (y0 + 1e-3)
                for b in range(NB):
                    if np.digitize([b * 5 + 2.5], [40, 60, 75])[0] == bb: v = base[b] * mult; self.PO[s, b] = v / v.sum()
        T = K[K.c == 0].copy(); T['d6'] = np.clip(T.sd0 + 6, -20, 20).astype(int); T['h2'] = (T.half == 2).astype(int)
        self.TRY = np.zeros((2, 41, 3))   # P(7, 6, 8) after a TD by half and the offense's lead after the 6 (1st half pooled)
        for hh in range(2):
            for d in range(-20, 21):
                x = T[(T.h2 == hh) & T.d6.between(d - (0 if hh else 30), d + (0 if hh else 30))].off_pts
                c = np.array([(x == 7).sum(), (x == 6).sum(), (x == 8).sum()], float) + np.array([0.92, 0.04, 0.04]) * 3; self.TRY[hh, d + 20] = c / c.sum()
        self.NS = [K[(K.c == c) & K.next_y.notna()].next_y.values for c in range(6)]
        self.SEC = [[(lambda v: v if len(v) >= 20 else K[K.c == c].sec.values)(K[(K.s == s) & (K.c == c)].sec.values) for c in range(7)] for s in range(NST)]
        self.ret_rate = float((D.def_pts > 0).mean()); self.ko = K[K.c == 0].next_y.dropna().values
        O = pd.read_parquet(drives); O = O[O.season.between(2023, 2025) & (O.period == 5)]
        self.OTP = np.array([(O.off_pts >= 6).mean(), (O.off_pts == 3).mean(), (O.off_pts == 0).mean()])   # 1st OT possession: TD, FG, none
        self.OTP /= self.OTP.sum()
        self.grid = np.linspace(-2, 2, 21); self.look = np.array([self.game(np.full(6000, t), np.zeros(6000))[0].mean() for t in self.grid])

    def outcome(self, b, th, s):
        p = self.PO[s, b].copy(); p[:, 0] *= np.exp(th); p[:, 1] *= np.exp(0.6 * th); p /= p.sum(1, keepdims=True)
        return (self.rng.random(len(b))[:, None] > np.cumsum(p, 1)).sum(1).clip(0, 6)

    def game(self, th_h, th_a, tempo=1.0, rec=None):
        """rec: optional list; each drive step appends its arrays (for the Watch-the-sim replays, models/sims/build.py); an
        end-of-half drive is recorded as c=2 with done=True. Recording draws no random numbers."""
        r = self.rng; n = len(th_h); hs = np.zeros(n); as_ = np.zeros(n); first = (r.random(n) < 0.5).astype(int)
        for half in (1, 2):
            poss = first.copy() if half == 1 else 1 - first
            y = r.choice(self.ko, n); clock = np.full(n, 1800.0); live = np.ones(n, bool)
            for _ in range(34):
                if not live.any(): break
                th = np.where(poss == 0, th_h, th_a); lead = np.where(poss == 0, hs - as_, as_ - hs)
                s = state(np.full(n, half == 2), clock, lead)
                c = self.outcome((y // 5).clip(0, NB - 1).astype(int), th, s)
                sec = np.zeros(n)
                for k in range(7):
                    for ss in range(NST):
                        m = (c == k) & (s == ss)
                        if m.any(): sec[m] = r.choice(self.SEC[ss][k], m.sum())
                sec *= tempo
                over = sec > clock; frac = np.where(over, clock / np.maximum(sec, 1), 1.0)
                c = np.where(over & (r.random(n) > frac), 6, c); done = over | (c == 6)
                d6 = (np.clip(lead + 6, -20, 20) + 20).astype(int); pr = self.TRY[half - 1, d6]; u = r.random(n)
                tdp = np.where(u < pr[:, 0], 7, np.where(u < pr[:, 0] + pr[:, 1], 6, 8))
                pts = np.where(c == 0, tdp, np.where(c == 1, 3, 0)).astype(float)
                ret = (r.random(n) < self.ret_rate) & ((c == 4) | (c == 2))
                add_off = pts * live; add_def = (ret * 7.0 + (c == 5) * 2) * live; oh = poss == 0
                if rec is not None: rec.append(dict(half=half, live=live.copy(), poss=poss.copy(), y=y.copy(), c=np.where(c == 6, 2, c), done=done.copy(), ret=ret.copy(), off=add_off, dfn=add_def, sec=np.minimum(sec, clock)))
                hs += np.where(oh, add_off, add_def); as_ += np.where(oh, add_def, add_off)
                clock = np.where(c == 6, 0.0, clock - sec); live = live & (clock > 0)
                nxt = np.choose(np.minimum(c, 5), [r.choice(self.NS[k], n) for k in range(6)]).astype(float)
                y = np.where(ret, r.choice(self.ko, n), nxt)
                poss = np.where(ret, poss, 1 - poss)   # return TD: the scorer kicks off; safety: the scorer gets the free kick
        self.overtime(th_h, th_a, hs, as_, rec)
        return hs, as_

    def overtime(self, th_h, th_a, hs, as_, rec=None):
        """College overtime (roadmap step 3, 2026-11-15): each round both teams get one possession from the 25 (TD / FG / no
        score from 2023-25 OT drives, tilted by team strength like regulation drives); the 2nd team needs a TD after a TD (goes
        for it on 4th down instead of kicking) and plays for the FG after a stop. Round 1 kicks the PAT, round 2 must go for 2
        after a TD, round 3+ is alternating 2-pt plays (45% each). Edits hs / as_ in place; replays get half=3 drives."""
        r = self.rng; n = len(th_h); tied = hs == as_; first = (r.random(n) < 0.5).astype(int); TWO = 0.45
        for rd in range(1, 11):
            if not tied.any(): break
            add = np.zeros((2, n)); order = first if rd % 2 else 1 - first
            for j in (0, 1):
                side = np.where(j == 0, order, 1 - order); th = np.where(side == 0, th_h, th_a)
                if rd >= 3:
                    pts = (r.random(n) < TWO) * 2.0; c = np.where(pts > 0, 0, 3)
                else:
                    p = np.tile(self.OTP, (n, 1)); p[:, 0] *= np.exp(th); p[:, 1] *= np.exp(0.6 * th)
                    if j == 1:
                        a0 = np.where(order == 0, add[0], add[1]); need_td = a0 >= 6; need_fg = a0 == 0
                        p[need_td, 0] += 0.4 * p[need_td, 1]; p[need_td, 2] += 0.6 * p[need_td, 1]; p[need_td, 1] = 0
                        p[need_fg, 1] += 0.5 * p[need_fg, 0]; p[need_fg, 0] *= 0.5
                    p /= p.sum(1, keepdims=True); c = (r.random(n)[:, None] > np.cumsum(p, 1)).sum(1).clip(0, 2)
                    td = np.where(rd == 1, np.where(r.random(n) < 0.97, 7, 6), np.where(r.random(n) < TWO, 8, 6))
                    pts = np.where(c == 0, td, np.where(c == 1, 3, 0)).astype(float); c = np.where(c == 2, 3, c)
                pts = pts * tied; add[0] += np.where(side == 0, pts, 0); add[1] += np.where(side == 1, pts, 0)
                if rec is not None: rec.append(dict(half=3, live=tied.copy(), poss=side.copy(), y=np.full(n, 25.0), c=c, done=np.zeros(n, bool), ret=np.zeros(n, bool),
                                                    off=pts, dfn=np.zeros(n), sec=np.zeros(n)))
            hs += add[0]; as_ += add[1]; tied = tied & (hs == as_)

    def run(self, hs, as_, n=2000, shrink=1.0, rec=None):
        """score sims for one game whose Model A means are hs/as_ (margin shrunk around the mid by `shrink`)"""
        mid, hm = (hs + as_) / 2, (hs - as_) / 2 * shrink
        c = self.rng.normal(0, SIGC, n)
        th = np.interp(mid + hm - BIAS + c + self.rng.normal(0, SIGI, n), self.look, self.grid)
        ta = np.interp(mid - hm - BIAS + c + self.rng.normal(0, SIGI, n), self.look, self.grid)
        return self.game(th, ta, rec=rec)
