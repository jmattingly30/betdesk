"""Grid tables of score distributions from the iid model: total-games pmf, margin pmf, set-score probs."""
import numpy as np, iid, sys
bo = int(sys.argv[1])
G = iid.GRID
TMAX = 40 if bo == 3 else 70
tot = np.zeros((len(G), len(G), TMAX)); mar = np.zeros((len(G), len(G), 2 * TMAX + 1)); win = np.zeros((len(G), len(G)))
sets = np.zeros((len(G), len(G), 2 * (bo // 2 + 1)))  # P(A wins need-k), k=0..need-1 ; then P(B wins need-k)
need = bo // 2 + 1
for i, a in enumerate(G):
    for j, b in enumerate(G):
        m = iid.match_dist(a, b, bo)
        win[i, j] = m['p']
        for k, v in m['total'].items():
            tot[i, j, min(k, TMAX - 1)] += v
        for k, v in m['margin'].items():
            mar[i, j, int(np.clip(k, -TMAX, TMAX)) + TMAX] += v
        for (x, y), v in m['sets'].items():
            if x == need: sets[i, j, y] += v
            else: sets[i, j, need + x] += v
np.savez_compressed(f'../data/dist_bo{bo}.npz', tot=tot, mar=mar, win=win, sets=sets)
print('done', bo)
