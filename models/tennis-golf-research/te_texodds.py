"""Men's model vs tennisexplorer openers (Pinnacle, bet365, best other book), graded vs the Pinnacle close.
Usage: PRED=../data/predx26c_atp.parquet te_texodds.py ../../op/tex_odds_men.jsonl [cats]"""
import os, sys, json, gzip
import tex_odds
import numpy as np, pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

def fair(oa, ob):
    """power de-vig -> prob of a"""
    if not (oa and ob and oa > 1 and ob > 1): return np.nan
    ia, ib = 1 / oa, 1 / ob
    if ia + ib <= 1: return ia / (ia + ib)
    k = brentq(lambda k: ia ** k + ib ** k - 1, 1, 10)
    return ia ** k

lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
ll = lambda y, p: -np.mean(y * np.log(np.clip(p, .01, .99)) + (1 - y) * np.log(np.clip(1 - p, .01, .99)))

RES = pd.read_csv('../../tex/results_all.csv', usecols=['mid', 'pid_a', 'pid_b', 'odds_a']).drop_duplicates('mid').set_index('mid')
rows = []
for l in open(sys.argv[1]):
    r = json.loads(l); B = r['books']
    if not B: continue
    pg = tex_odds.players(gzip.open(f"{tex_odds.MD}/{r['mid']}.html.gz", 'rt', errors='ignore').read())
    if int(r['mid']) not in RES.index or len(pg) < 2: continue
    rv = pg[0] != RES.loc[int(r['mid']), 'pid_a']   # page lists the loser first -> swap so a = winner (results order)
    if rv:
        for k in B: B[k] = dict(a=B[k]['b'], b=B[k]['a'])
    d = dict(fid='tex' + r['mid'], page_rev=rv, res_oa=RES.loc[int(r['mid']), 'odds_a'])
    for bk, tag in [('Pinnacle', 'pin'), ('bet365', 'b365')]:
        if bk in B:
            a, b = B[bk]['a'], B[bk]['b']
            d[tag + '_oa'], d[tag + '_ob'], d[tag + '_ca'], d[tag + '_cb'] = a['open'], b['open'], a['now'], b['now']
            d[tag + '_moved'] = bool(a['hist'] or b['hist'])
    # best opening price on each side across all books (what a shopper could take at open)
    oa = [B[k]['a']['open'] for k in B if B[k]['a']['open']]; ob = [B[k]['b']['open'] for k in B if B[k]['b']['open']]
    d['best_oa'], d['best_ob'], d['nbooks'] = (max(oa) if oa else np.nan), (max(ob) if ob else np.nan), len(B)
    rows.append(d)
o = pd.DataFrame(rows)
P = pd.read_parquet(os.environ.get('PRED', '../data/predx26c_atp.parquet'))
cats = sys.argv[2].split(',') if len(sys.argv) > 2 else ['Challenger', 'Tour']
x = P[P.cat.isin(cats)].merge(o, on='fid')
# orientation: page order should match results order (a = winner); check vs average odds oa/ob
print('matched', len(x))
for t in ['pin', 'b365']:
    x[t + '_o'] = [fair(a, b) for a, b in zip(x[t + '_oa'], x[t + '_ob'])]
    x[t + '_c'] = [fair(a, b) for a, b in zip(x[t + '_ca'], x[t + '_cb'])]
x = x[x.pin_o.notna() & x.pin_c.notna()].copy()
# books are in results order (a = winner); the pred row has its own a/b order -> flip books where pred a is the loser
sw = ~np.isclose(x.oa, x.res_oa) | x.res_oa.isna()
print('pred rows with loser as a', round(sw.mean(), 2), 'y mean', round(x.y.mean(), 2))
for c in ['pin_o', 'pin_c', 'b365_o', 'b365_c']:
    x.loc[sw, c] = 1 - x.loc[sw, c]
for a_, b_ in [('pin_oa', 'pin_ob'), ('b365_oa', 'b365_ob'), ('best_oa', 'best_ob'), ('pin_ca', 'pin_cb'), ('b365_ca', 'b365_cb')]:
    x.loc[sw, [a_, b_]] = x.loc[sw, [b_, a_]].values
x['y'] = x.y.astype(float)
x['day'] = x.start.astype(str).str[:10]
print('rows', len(x), 'Pinnacle moved', round(x.pin_moved.mean(), 2), 'bet365 present', round(x.b365_o.notna().mean(), 2))
for c in cats:
    s = x[x.cat == c]
    if not len(s): continue
    out = {k: round(ll(s.y, s[k]), 4) for k in ['p', 'pin_o', 'pin_c']}
    sb = s[s.b365_o.notna()]
    out['b365_o (own rows)'] = round(ll(sb.y, sb.b365_o), 4); out['pin_o same rows'] = round(ll(sb.y, sb.pin_o), 4)
    print(c, len(s), out)

def blend_test(s, ob='pin_o', mid_day=None):
    """fit blend weights on the first half of days, test on the second half (and vice versa); returns out-of-sample blended probs"""
    days = sorted(s.day.unique()); cut = days[len(days) // 2] if mid_day is None else mid_day
    s = s[s[ob].notna()].copy(); s['bl'] = np.nan
    for tr, te in [(s.day < cut, s.day >= cut), (s.day >= cut, s.day < cut)]:
        X = np.c_[lg(s[ob]), lg(s.p)]
        m = LogisticRegression(C=1e4).fit(X[tr], s.y[tr])
        s.loc[te, 'bl'] = m.predict_proba(X[te])[:, 1]
        print(f'   {ob} fit on {int(tr.sum())}: w_open {m.coef_[0][0]:.2f} w_model {m.coef_[0][1]:.2f}')
    return s

for c in cats:
    s = x[x.cat == c]
    if len(s) < 100: continue
    print('==', c)
    for ob, price_a, price_b in [('pin_o', 'pin_oa', 'pin_ob'), ('b365_o', 'b365_oa', 'b365_ob'), ('pin_o', 'best_oa', 'best_ob')]:
        b = blend_test(s, ob)
        print(f'  {ob} -> bets at {price_a[:-1]}: LL open {ll(b.y, b[ob]):.4f} blend {ll(b.y, b.bl):.4f} close {ll(b.y, b.pin_c):.4f}')
        for edge in [.03, .06, .10]:
            res = []
            for side in ['a', 'b']:
                q = b.bl if side == 'a' else 1 - b.bl
                pr = b[price_a] if side == 'a' else b[price_b]
                yy = b.y if side == 'a' else 1 - b.y
                cl = b.pin_c if side == 'a' else 1 - b.pin_c
                f = (q * pr - 1 >= edge) & (pr <= 4)
                res.append(pd.DataFrame(dict(ret=np.where(yy[f] == 1, pr[f] - 1, -1.), clv=cl[f] * pr[f] - 1)))
            r = pd.concat(res)
            if len(r): print(f'     edge {edge:.0%}: bets {len(r)} ROI {r.ret.mean():+.3f} CLV {r.clv.mean():+.3f} beat close {np.mean(r.clv > 0):.2f}')
x.to_parquet('../data/texodds_men_test.parquet')
