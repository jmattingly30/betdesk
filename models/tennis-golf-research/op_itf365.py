"""2026 opener test at the books J can use: our live blend (model + market rating from past closes) vs DraftKings
and FanDuel OPENING prices, graded by results and by CLV vs Pinnacle's no-vig close (OddsPapi histories)."""
import json, sys, os
import numpy as np, pandas as pd
from te_vs_mkt import fair
cats = sys.argv[1].split(',') if len(sys.argv) > 1 else ['ITF Women']
# our live model (participant-1 orientation) and the 2025-holdout blend weights: logit p = wo*logit(open fair) + wm*logit(model)
from bridge import short_key
X = pd.concat([pd.read_parquet(f'../data/predx26{os.environ.get("PRED_V", "")}_{t}.parquet') for t in ('atp', 'wta')])
X['day'] = X.start.dt.tz_convert(None).dt.normalize()
PX = {}; NX = {}; NG = {}
for r in X.itertuples(index=False):
    ka, kb = short_key(r.a), short_key(r.b)
    PX[(ka, kb, r.day)] = r.p; PX[(kb, ka, r.day)] = 1 - r.p; NX[(ka, kb, r.day)] = NX[(kb, ka, r.day)] = r.gelo_n
O = pd.read_parquet('../data/op_results_2026.parquet')
P = {}
for r in O.itertuples(index=False):
    ka, kb = short_key(r.an), short_key(r.bn); d = r.start.tz_convert(None).normalize()
    for dd in (0, -1, 1):
        v = PX.get((ka, kb, d + pd.Timedelta(days=dd)))
        if v is not None: P[r.fid] = v; NG[r.fid] = NX[(ka, kb, d + pd.Timedelta(days=dd))]; break
W = {'ITF Women': (0.749, 0.43), 'Challenger': (0.798, 0.198), 'WTA': (0.703, 0.296), 'ATP': (0.82, 0.096), 'WTA 125K': (0.703, 0.296)}
lg = lambda p: np.log(p / (1 - p))
rows = []
for l in open('../../op/' + os.environ.get('OP_HIST', 'hist.jsonl')):
    r = json.loads(l)
    if 'books' not in r or r.get('cat') not in cats or r['fid'] not in P: continue
    bk = r['books']
    pin = bk.get('pinnacle', {}).get('121', {}).get('o', {})
    if not (pin.get('1', {}).get('close') and pin.get('2', {}).get('close')): continue
    fc = fair(np.array([pin['1']['close'][1]]), np.array([pin['2']['close'][1]]))[0]
    b3 = bk.get('bet365', {}).get('121', {}).get('o', {})
    f3 = fair(np.array([b3['1']['close'][1]]), np.array([b3['2']['close'][1]]))[0] if (b3.get('1', {}).get('close') and b3.get('2', {}).get('close')) else np.nan
    win = None
    for b in ('pinnacle', 'bet365', 'draftkings', 'fanduel'):
        o = bk.get(b, {}).get('121', {}).get('o', {})
        if o.get('1', {}).get('last') and o.get('2', {}).get('last'):
            a, c = o['1']['last'][1], o['2']['last'][1]
            if a <= 1.06 and c >= 6: win = 1; break
            if c <= 1.06 and a >= 6: win = 0; break
    for b in ('bet365', 'draftkings', 'fanduel', 'pinnacle'):
        o = bk.get(b, {}).get('121', {}).get('o', {})
        if not (o.get('1', {}).get('series') and o.get('2', {}).get('series')): continue
        s1, s2 = o['1']['series'], o['2']['series']
        t0 = max(s1[0][0], s2[0][0])  # first moment both sides are posted
        pa = [x[1] for x in s1 if x[0] <= t0][-1]; pb = [x[1] for x in s2 if x[0] <= t0][-1]
        fo = fair(np.array([pa]), np.array([pb]))[0]; wo, wm = W[r['cat']]
        p = 1 / (1 + np.exp(-(wo * lg(fo) + wm * lg(np.clip(P[r['fid']], .01, .99)))))
        rows.append(dict(fid=r['fid'], book=b, open_h=-t0 / 3600, pa=pa, pb=pb, p=p, fo=fo, fc=fc, f3=f3, win=win))
D = pd.DataFrame(rows)
print('matches', D.fid.nunique(), D.groupby('book').size().to_dict(), 'median hours before start of open', D.groupby('book').open_h.median().round(1).to_dict())
out = []
for b, z in D.groupby('book'):
    for th in (0.0, 0.03, 0.06):
        e1 = z.p * z.pa - 1; e2 = (1 - z.p) * z.pb - 1
        sel1, sel2 = (e1 > th) & z.pa.between(1.2, 5), (e2 > th) & z.pb.between(1.2, 5)
        clv = np.r_[(z.fc * z.pa - 1)[sel1], ((1 - z.fc) * z.pb - 1)[sel2]]
        clv3 = np.r_[(z.f3 * z.pa - 1)[sel1], ((1 - z.f3) * z.pb - 1)[sel2]]
        ret = np.r_[np.where(z.win[sel1] == 1, z.pa[sel1] - 1, np.where(z.win[sel1].isna(), np.nan, -1.0)),
                    np.where(z.win[sel2] == 0, z.pb[sel2] - 1, np.where(z.win[sel2].isna(), np.nan, -1.0))]
        out.append(dict(book=b, edge_over=th, bets=len(clv), clv=clv.mean() if len(clv) else np.nan, clv_b365close=np.nanmean(clv3) if len(clv3) else np.nan, clv_pos=(clv > 0).mean() if len(clv) else np.nan,
                        roi=np.nanmean(ret) if len(ret) else np.nan, se=np.nanstd(ret) / np.sqrt(max(np.sum(~np.isnan(ret)), 1))))
print(pd.DataFrame(out).round(4).to_string())
# opener accuracy: log loss of each book's opener vs blend, completed with known winner
L = D[D.win.notna()]
for b, z in L.groupby('book'):
    y = z.win.values; ll = lambda p: (lambda q: -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))(np.clip(p, .01, .99))
    print(b, 'n', len(z), 'LL open', round(ll(z.fo.values), 4), 'blend', round(ll(z.p.values), 4), 'pin close', round(ll(z.fc.values), 4))
hold = D.assign(h=1 / D.pa + 1 / D.pb - 1).groupby('book').h.median().round(3).to_dict(); print('median opener hold', hold)
# diagnosis: refit blend weights on 2026 rows (in-sample) per book, and by month split (fit Sep, test Oct)
from sklearn.linear_model import LogisticRegression
D['lm'] = lg(np.clip(D.p, .01, .99))  # placeholder, recomputed below from model prob
D['pm'] = [P[f] for f in D.fid]
D['mon'] = [0] * len(D)
for b, z in D[D.win.notna()].groupby('book'):
    X = np.c_[lg(np.clip(z.fo, .01, .99)), lg(np.clip(z.pm, .01, .99))]
    l = LogisticRegression(C=100).fit(np.r_[X, -X], np.r_[z.win, 1 - z.win])
    print(b, 'refit 2026 weights open/model', l.coef_[0].round(3))
