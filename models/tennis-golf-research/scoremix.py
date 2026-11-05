"""Score model v2: the serve-point gap varies day to day (random 'form' effect), so the match distribution is a
mixture over gaps. mean gap is solved so the mixture's win probability equals the calibrated model probability.
Tune form sd (sigma) and hold-level shift on 2015-2018, grade on 2019-2024."""
import sys
import numpy as np, pandas as pd
import iid
from model import ll

tour = sys.argv[1] if len(sys.argv) > 1 else 'atp'
G = iid.GRID
D = {bo: np.load(f'../data/dist_bo{bo}.npz') for bo in (3, 5)}
Z, W = np.polynomial.hermite_e.hermegauss(9); W = W / W.sum()


def interp(T, pa, pb):
    pa = np.clip(pa, G[0], G[-1] - 1e-6); pb = np.clip(pb, G[0], G[-1] - 1e-6)
    fa = (pa - G[0]) / 0.005; fb = (pb - G[0]) / 0.005
    i = np.floor(fa).astype(int); j = np.floor(fb).astype(int)
    da = (fa - i)[..., None] if T.ndim == 3 else fa - i
    db = (fb - j)[..., None] if T.ndim == 3 else fb - j
    return T[i, j] * (1 - da) * (1 - db) + T[i + 1, j] * da * (1 - db) + T[i, j + 1] * (1 - da) * db + T[i + 1, j + 1] * da * db


def mix(s, p, bo, sigma, shift=0.0):
    s = s + shift
    T = D[bo]
    def winp(mu):
        return sum(w * interp(T['win'], s + (mu + sigma * z) / 2, s - (mu + sigma * z) / 2) for z, w in zip(Z, W))
    lo, hi = np.full_like(s, -0.5), np.full_like(s, 0.5)
    for _ in range(35):
        m = (lo + hi) / 2; q = winp(m)
        lo = np.where(q < p, m, lo); hi = np.where(q >= p, m, hi)
    mu = (lo + hi) / 2
    out = {}
    for key in ('tot', 'mar', 'sets'):
        out[key] = sum(w * interp(T[key], s + (mu + sigma * z) / 2, s - (mu + sigma * z) / 2) for z, w in zip(Z, W))
    return out


F = pd.read_parquet(f'../data/feat_{tour}.parquet').join(pd.read_parquet(f'../data/pred_{tour}.parquet'))
x = F[(F.date >= '2015-01-01') & (~F.ret) & F.p_model.notna() & (F.vs_a < 0.2) & (F.vs_b < 0.2) & F.games_w.notna() & (F.games_w > 0)].copy()
x['bo'] = x.bo.fillna(3).astype(int)
x['s'] = (x.pa + x.pb) / 2
x['tg'] = x.games_w + x.games_l
x['marA'] = np.where(x.y == 1, x.games_w - x.games_l, x.games_l - x.games_w)
x['straight'] = (x.sets_l == 0).astype(int)
x['yr'] = x.date.dt.year


def grade(z, bo, sigma, shift):
    o = mix(z.s.values, z.p_model.values, bo, sigma, shift)
    TM = o['tot'].shape[1]; k = np.arange(TM); half = (o['mar'].shape[1] - 1) // 2
    tg = np.minimum(z.tg.values.astype(int), TM - 1)
    line = 22.5 if bo == 3 else 38.5
    p_over = o['tot'][:, k > line].sum(1)
    need = bo // 2 + 1
    p_str = o['sets'][:, 0] + o['sets'][:, need]
    fav = z.p_model.values >= 0.5
    mk = np.arange(-half, half + 1)
    pA_cover_m35 = o['mar'][:, mk > 3.5].sum(1); pB_cover_m35 = o['mar'][:, mk < -3.5].sum(1)
    p_cov = np.where(fav, pA_cover_m35, pB_cover_m35)
    cov = np.where(fav, z.marA.values > 3.5, z.marA.values < -3.5).astype(int)
    mean_t = (o['tot'] * k).sum(1)
    lp_t = np.log(np.clip(o['tot'][np.arange(len(z)), tg], 1e-6, 1))
    over = (z.tg.values > line).astype(int)
    return dict(n=len(z), bias=round((mean_t - z.tg.values).mean(), 3), mae=round(np.abs(mean_t - z.tg.values).mean(), 3),
                ll_total_pmf=round(-lp_t.mean(), 4), ll_over=round(ll(p_over, over), 4), over_pred=round(p_over.mean(), 3), over_act=round(over.mean(), 3),
                ll_straight=round(ll(p_str, z.straight.values), 4), str_pred=round(p_str.mean(), 3), str_act=round(z.straight.mean(), 3),
                ll_cover=round(ll(p_cov, cov), 4), cov_pred=round(p_cov.mean(), 3), cov_act=round(cov.mean(), 3)), (p_over, over, p_str, p_cov, cov, mean_t)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for bo in (3, 5):
        tr = x[(x.bo == bo) & (x.yr <= 2018)]
        if len(tr) > 40000:
            tr = tr.sample(40000, random_state=1)
        res = []
        for sigma in (0.0, 0.06, 0.07, 0.08, 0.09, 0.10):
            for shift in (0.0, -0.01, -0.02, -0.03):
                r, _ = grade(tr, bo, sigma, shift); r.update(sigma=sigma, shift=shift); res.append(r)
        R = pd.DataFrame(res)
        R['score'] = R.ll_total_pmf + R.ll_straight + R.ll_cover
        print(f'--- best of {bo} tuning 2015-18'); print(R.sort_values('score').head(6).to_string())
        b = R.sort_values('score').iloc[0]
        te = x[(x.bo == bo) & (x.yr >= 2019)]
        r0, _ = grade(te, bo, 0.0, 0.0)
        r1, extra = grade(te, bo, b['sigma'], b['shift'])
        print(f'--- best of {bo} TEST 2019-24, no form (v1):'); print(r0)
        print(f'--- best of {bo} TEST 2019-24, form sd {b['sigma']} shift {b['shift']}:'); print(r1)
        # constant baselines on test
        print('  const over logloss', round(ll(np.full(len(te), (te.tg > (22.5 if bo == 3 else 38.5)).mean()), (te.tg > (22.5 if bo == 3 else 38.5)).astype(int).values), 4),
              'const straight logloss', round(ll(np.full(len(te), te.straight.mean()), te.straight.values), 4))
        p_over, over, p_str, p_cov, cov, mean_t = extra
        cal = pd.DataFrame({'p': p_over, 'y': over}); cal['bk'] = pd.cut(cal.p, [0, .3, .4, .5, .6, .7, 1])
        print(cal.groupby('bk', observed=True).agg(n=('y', 'size'), pred=('p', 'mean'), act=('y', 'mean')).round(3).to_string())
        te = te.assign(mean_t=mean_t, p_over=p_over, p_cov=p_cov, cov=cov, p_str=p_str)
        te.to_parquet(f'../data/scoremix_{tour}_bo{bo}.parquet')
