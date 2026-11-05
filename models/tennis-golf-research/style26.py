"""Prior-only per-player scoring style from tex scores: games per set, tiebreak-set rate, hold proxy (games won share in sets lost).
Output ../data/style26.parquet keyed by fid with *_w (results winner) and *_l columns. Shrunk toward tour means (k sets)."""
import re
import numpy as np, pandas as pd
R = pd.read_csv('../../tex/results_all.csv').drop_duplicates('mid').sort_values('day')
R['tour'] = np.where(R.ev.str.contains('wta-women'), 'w', 'm')
K = 20
st = {}
mu = {'w': (9.6, .10), 'm': (10.0, .16)}
out = []
for r in R.itertuples():
    sets = re.findall(r'(\d+)-(\d+)', str(r.score))
    sets = [(int(a), int(b)) for a, b in sets if int(a) + int(b) <= 13 and max(int(a), int(b)) >= 6]
    m = mu[r.tour]; row = dict(fid='tex' + str(r.mid))
    for side, pid in [('w', r.pid_a), ('l', r.pid_b)]:
        s = st.get(pid, [0, 0., 0.])
        row[f'gps_{side}'] = (s[1] + K * m[0]) / (s[0] + K)
        row[f'tb_{side}'] = (s[2] + K * m[1]) / (s[0] + K)
        row[f'ns_{side}'] = s[0]
    out.append(row)
    for pid in (r.pid_a, r.pid_b):
        s = st.setdefault(pid, [0, 0., 0.])
        for a, b in sets:
            s[0] += 1; s[1] += a + b; s[2] += (a + b == 13)
pd.DataFrame(out).to_parquet('../data/style26.parquet')
print(len(out))
