"""CFB Model A shared pieces (J 2026-11-05 rollout; fourth sport on the two-model system after CBB, NFL, NBA).
- v4 means: the tested CFB rebuild terms on top of the desk's v3 (unblended): program SRS (s1, s23), coach career win%
  (c_cwin), crowd size (x_att); weeks 1-3 also keep v3's roster terms (RT + D). Total = v3 total + in-season tempo (x_ip).
  Same as models/rebuild/football/cfb_gap/live/cfb_live.py (which writes app/data/cfb_live.json for live games).
- drive simulator (research/cfb-sim-2026-10/drive_sim.py, J yardstick regrade 2026-10-31): league drive tables 2023-25,
  TD/FG odds tilted so each team's expected points match Model A, shared game shock 3.9 + team shock 2.0, bias .56, margin
  slope (SHRINK) fitted on 2014-21. Used for ranges, win %, team-total over chances, key numbers. Means stay Model A's.
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
SIGC, SIGI, BIAS = 3.9, 2.0, 0.56
CAT = {'TD': 0, 'FG': 1, 'PUNT': 2, 'PUNT TD': 2, 'PUNT RETURN TD': 2, 'DOWNS': 3, 'DOWNS TD': 3, 'MISSED FG': 3, 'MISSED FG TD': 3,
       'INT': 4, 'FUMBLE': 4, 'INT TD': 4, 'FUMBLE RETURN TD': 4, 'FUMBLE TD': 4, 'SF': 5}
NB = 20; PAT = np.array([6, 7, 8]); PATP = np.array([0.04, 0.92, 0.04])

class Sim:
    def __init__(self, drives=f"{HERE}/data/drives_2023_25.parquet", seed=11):
        self.rng = np.random.default_rng(seed)
        D = pd.read_parquet(drives); D = D[D.season.between(2023, 2025) & (D.period <= 4) & D.y0.between(1, 99)].copy().sort_values(['game_id', 'n'])
        D['c'] = D.res.map(CAT); D['half'] = np.where(D.period <= 2, 1, 2)
        D['next_y'] = D.groupby(['game_id', 'half']).y0.shift(-1); D['sec'] = (D.t0 - D.t1).clip(5, 600)
        K = D[D.c.notna()].copy(); K['c'] = K.c.astype(int); K['b'] = (K.y0 // 5).clip(0, NB - 1).astype(int)
        self.PO = np.zeros((NB, 6))
        for b in range(NB):
            v = K[K.b == b].c.value_counts().reindex(range(6), fill_value=0).values + 0.5; self.PO[b] = v / v.sum()
        self.NS = [K[(K.c == c) & K.next_y.notna()].next_y.values for c in range(6)]
        self.SEC = [K[K.c == c].sec.values for c in range(6)]
        self.ret_rate = float((D.def_pts > 0).mean()); self.ko = K[K.c == 0].next_y.dropna().values
        self.grid = np.linspace(-2, 2, 21); self.look = np.array([self.game(np.full(6000, t), np.zeros(6000))[0].mean() for t in self.grid])

    def outcome(self, b, th):
        p = self.PO[b].copy(); p[:, 0] *= np.exp(th); p[:, 1] *= np.exp(0.6 * th); p /= p.sum(1, keepdims=True)
        return (self.rng.random(len(b))[:, None] > np.cumsum(p, 1)).sum(1).clip(0, 5)

    def game(self, th_h, th_a, tempo=1.0):
        r = self.rng; n = len(th_h); hs = np.zeros(n); as_ = np.zeros(n)
        for half in (1, 2):
            poss = (r.random(n) < 0.5).astype(int) if half == 1 else np.ones(n, int)
            y = r.choice(self.ko, n); clock = np.full(n, 1800.0); live = np.ones(n, bool)
            for _ in range(30):
                if not live.any(): break
                th = np.where(poss == 0, th_h, th_a); c = self.outcome((y // 5).clip(0, NB - 1).astype(int), th)
                sec = np.choose(c, [r.choice(self.SEC[k], n) for k in range(6)]) * tempo
                done = sec > clock; frac = np.where(done, clock / np.maximum(sec, 1), 1.0)
                c = np.where(done & (r.random(n) > frac), 2, c)
                pts = np.where(c == 0, r.choice(PAT, n, p=PATP), np.where(c == 1, 3, 0)).astype(float)
                ret = (r.random(n) < self.ret_rate) & ((c == 4) | (c == 2))
                add_off = pts * live; add_def = (ret * 7.0 + (c == 5) * 2) * live; oh = poss == 0
                hs += np.where(oh, add_off, add_def); as_ += np.where(oh, add_def, add_off)
                clock = clock - sec; live = live & (clock > 0)
                y = np.where(ret, r.choice(self.ko, n), np.choose(c, [r.choice(self.NS[k], n) for k in range(6)]).astype(float)); poss = 1 - poss
        return hs, as_

    def run(self, hs, as_, n=2000, shrink=1.0):
        """score sims for one game whose Model A means are hs/as_ (margin shrunk around the mid by `shrink`)"""
        mid, hm = (hs + as_) / 2, (hs - as_) / 2 * shrink
        c = self.rng.normal(0, SIGC, n)
        th = np.interp(mid + hm - BIAS + c + self.rng.normal(0, SIGI, n), self.look, self.grid)
        ta = np.interp(mid - hm - BIAS + c + self.rng.normal(0, SIGI, n), self.look, self.grid)
        return self.game(th, ta)
