"""PGA round-score model for head-to-head round matchups (walk-forward, 2010-2026 ESPN hole data).

score(i, round r) = course_day(r) + skill_i + e,  e ~ N(0, sd_i)
skill is a Kalman state (drifts between rounds), course_day is the field-strength-corrected round mean.
H2H in the same round: diff = skill_b - skill_a (+ extras), integer strokes -> P(A wins), P(tie), P(B wins).
"""
import sys, math
import numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm

R = pd.read_parquet('rounds.parquet')
R = R[R.holes == 18].copy()
R['round'] = pd.to_numeric(R['round'])
R['tid'] = pd.to_numeric(R.tournament_id)
R = R.sort_values(['year', 'tid', 'round']).reset_index(drop=True)

P = dict(v0=1.5, q=0.004, obs=2.85 ** 2, prior=1.0, sdk=0.15, ch_w=0.0, mom=0.0)
if len(sys.argv) > 1:
    for kv in sys.argv[1:]:
        k, v = kv.split('='); P[k] = float(v)

mu = defaultdict(lambda: P['prior'])     # skill = strokes WORSE than an average PGA field player (lower better)
v = defaultdict(lambda: P['v0'])
nr = defaultdict(int)
ss = defaultdict(float); sn = defaultdict(int)   # residual sum of squares for player sd
ch = defaultdict(float); chn = defaultdict(int)  # course history residuals (player, tournament_name)
last_res = {}
rows = []
rng = np.random.default_rng(3)
for (yr, tid, rd), g in R.groupby(['year', 'tid', 'round'], sort=False):
    ids = g.player_id.values
    sc = g.score.values.astype(float)
    tname = g.tournament_name.iloc[0]
    m_pre = np.array([mu[i] for i in ids]); v_pre = np.array([v[i] for i in ids])
    day = sc.mean() - m_pre.mean()                    # course-day effect, field-strength corrected
    res = sc - day - m_pre
    # player sd estimates (shrunk to pooled)
    sd_i = np.array([math.sqrt((ss[i] + 10 * P['obs']) / (sn[i] + 10)) for i in ids])
    chist = np.array([ch[(i, tname)] / (chn[(i, tname)] + 4) for i in ids])
    mom = np.array([last_res.get((i, tid), 0.0) for i in ids]) if rd > 1 else np.zeros(len(ids))
    # sample pairs for evaluation: pairs with close skill (like book matchups)
    if len(ids) >= 10 and yr >= 2013:
        order = rng.permutation(len(ids))
        for k in range(0, len(order) - 1, 2):
            a, b = order[k], order[k + 1]
            if nr[ids[a]] < 20 or nr[ids[b]] < 20:
                continue
            if rng.random() < 0.5:
                a, b = b, a
            rows.append(dict(year=yr, tid=tid, rd=rd, a=ids[a], b=ids[b], na=nr[ids[a]], nb=nr[ids[b]],
                             mu_a=m_pre[a], mu_b=m_pre[b], v_a=v_pre[a], v_b=v_pre[b], sd_a=sd_i[a], sd_b=sd_i[b],
                             ch_a=chist[a], ch_b=chist[b], mom_a=mom[a], mom_b=mom[b], s_a=sc[a], s_b=sc[b],
                             day=day))
    # update
    for j, i in enumerate(ids):
        vv = v[i] + P['q']
        k = vv / (vv + P['obs'])
        mu[i] += k * res[j]
        v[i] = vv * (1 - k)
        nr[i] += 1
        ss[i] += res[j] ** 2; sn[i] += 1
        ch[(i, tname)] += res[j]; chn[(i, tname)] += 1
        last_res[(i, tid)] = res[j]
E = pd.DataFrame(rows)
E.to_parquet('golf_pairs.parquet')


def probs(E, use_sd=False, ch_w=0.0, mom_w=0.0, sd_pool=2.85):
    d = (E.mu_b - E.mu_a) + ch_w * (E.ch_b - E.ch_a) + mom_w * (E.mom_b - E.mom_a)  # positive -> A better
    if use_sd:
        s = np.sqrt(E.sd_a ** 2 + E.sd_b ** 2 + E.v_a + E.v_b)
    else:
        s = np.sqrt(2 * sd_pool ** 2 + E.v_a + E.v_b)
    pa = 1 - norm.cdf((0.5 - d) / s)       # B - A >= 1 stroke  => A wins (A lower score)
    pb = norm.cdf((-0.5 - d) / s)
    pt = 1 - pa - pb
    return pa, pt, pb


def ll3(E, pa, pt, pb):
    o = np.sign(E.s_b - E.s_a)  # +1 A wins
    p = np.where(o > 0, pa, np.where(o < 0, pb, pt))
    return -np.mean(np.log(np.clip(p, 1e-6, 1)))


if __name__ == '__main__':
    print('pairs', len(E), 'tie rate', round((E.s_a == E.s_b).mean(), 4))
    base_p = 1 / 3
    o = np.sign(E.s_b - E.s_a)
    emp = [(o > 0).mean(), (o == 0).mean(), (o < 0).mean()]
    print('naive (no skill) 3-way logloss', round(-np.mean(np.log(np.where(o > 0, emp[0], np.where(o < 0, emp[2], emp[1])))), 4))
    for sdp in (2.6, 2.75, 2.9, 3.05):
        print('pooled sd', sdp, round(ll3(E, *probs(E, sd_pool=sdp)), 5))
    print('player sd', round(ll3(E, *probs(E, use_sd=True)), 5))
    for cw in (0.25, 0.5, 1.0):
        print('course history w', cw, round(ll3(E, *probs(E, ch_w=cw)), 5))
    for mw in (0.05, 0.1, 0.2):
        print('momentum w', mw, round(ll3(E, *probs(E, mom_w=mw)), 5))
    pa, pt, pb = probs(E)
    E['pa'], E['pt'] = pa, pt
    E['bk'] = pd.cut(E.pa / (E.pa + pb), [0, .4, .45, .5, .55, .6, 1])
    print(E.groupby('bk', observed=True).apply(lambda g: pd.Series({'n': len(g), 'pred_A_no_tie': (g.pa / (1 - g.pt)).mean(),
          'act_A_no_tie': ((g.s_a < g.s_b).sum() / max((g.s_a != g.s_b).sum(), 1)), 'pred_tie': g.pt.mean(), 'act_tie': (g.s_a == g.s_b).mean()})).round(4))
    E['gap'] = (E.mu_a - E.mu_b).abs()
    for lo, hi in ((0, .25), (.25, .5), (.5, 1), (1, 9)):
        z = (E.gap >= lo) & (E.gap < hi)
        o2 = o[z]; emp2 = [(o2 > 0).mean(), (o2 == 0).mean(), (o2 < 0).mean()]
        nv = -np.mean(np.log(np.where(o2 > 0, emp2[0], np.where(o2 < 0, emp2[2], emp2[1]))))
        print(f'gap {lo}-{hi}: n={z.sum()} model {ll3(E[z], pa[z], pt[z], pb[z]):.4f} naive {nv:.4f} better-player win% (no ties) {((E.s_a < E.s_b) == (E.mu_a < E.mu_b))[z & (E.s_a != E.s_b)].mean():.3f}')
    print('skill gap abs mean', round((E.mu_a - E.mu_b).abs().mean(), 3))
