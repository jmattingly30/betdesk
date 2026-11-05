"""2026 walk-forward blend: weights for logit(opener fair) + logit(model) fit on earlier months only (per book),
then bets at that book's opener. Graded by CLV vs Pinnacle and bet365 closes and by results."""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')
os.environ.setdefault('OP_HIST', 'hist365.jsonl')
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import op_itf365 as M
D = M.D.copy(); lg = M.lg
D['pm'] = [M.P[f] for f in D.fid]
import json
st = {}
for l in open('../../op/' + os.environ['OP_HIST']):
    r = json.loads(l)
    if 'start' in r: st[r['fid']] = r['start'][:7]
D['mon'] = D.fid.map(st)
D = D[D.fo.between(0.02, 0.98)]
D['ng'] = D.fid.map(M.NG)
if os.environ.get('MIN_NG'): D = D[D.ng >= int(os.environ['MIN_NG'])]
rows = []
for b, z in D.groupby('book'):
    for m in sorted(z.mon.unique()):
        tr = z[(z.mon < m) & z.win.notna()]; te = z[z.mon == m]
        if len(tr) < int(os.environ.get('MIN_TR', 300)): continue
        X = np.c_[lg(tr.fo), lg(np.clip(tr.pm, .01, .99))]
        l = LogisticRegression(C=100).fit(np.r_[X, -X], np.r_[tr.win, 1 - tr.win])
        p = l.predict_proba(np.c_[lg(te.fo), lg(np.clip(te.pm, .01, .99))])[:, 1]
        rows.append(te.assign(pw=p, wo=l.coef_[0][0], wm=l.coef_[0][1]))
W = pd.concat(rows)
print('test months', sorted(W.mon.unique()), 'weights', W.groupby('book')[['wo', 'wm']].mean().round(3).to_dict())
y = W.win
for b, z in W[W.win.notna()].groupby('book'):
    ll = lambda p: -np.mean(z.win * np.log(p) + (1 - z.win) * np.log(1 - p))
    print(b, 'n', len(z), 'LL open', round(ll(z.fo), 4), 'wf blend', round(ll(z.pw), 4), 'pin close', round(ll(z.fc.clip(.01, .99)), 4))
out = []
for b, z in W.groupby('book'):
    for th in (0.0, 0.02, 0.04):
        e1 = z.pw * z.pa - 1; e2 = (1 - z.pw) * z.pb - 1
        s1, s2 = (e1 > th) & z.pa.between(1.2, 5), (e2 > th) & z.pb.between(1.2, 5)
        clv = np.r_[(z.fc * z.pa - 1)[s1], ((1 - z.fc) * z.pb - 1)[s2]]
        c3 = np.r_[(z.f3 * z.pa - 1)[s1], ((1 - z.f3) * z.pb - 1)[s2]]
        ret = np.r_[np.where(z.win[s1].isna(), np.nan, np.where(z.win[s1] == 1, z.pa[s1] - 1, -1.0)), np.where(z.win[s2].isna(), np.nan, np.where(z.win[s2] == 0, z.pb[s2] - 1, -1.0))]
        out.append(dict(book=b, edge_over=th, bets=len(clv), clv_pin=np.nanmean(clv), clv_b365=np.nanmean(c3), roi=np.nanmean(ret), se=np.nanstd(ret) / np.sqrt(max(np.sum(~np.isnan(ret)), 1))))
print(pd.DataFrame(out).round(4).to_string())
