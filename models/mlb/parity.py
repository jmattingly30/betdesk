"""Parity check (2026-11-25): re-run J's app on the desk for the 2026 World Series days J's live Apps Script graded,
and compare eff with the pregame projection his app saved then (models/grading/history/MLB_FINALS.json.gz).
The season was over by then, so the stats behind both runs should be the same. Usage: python3 parity.py slate1.json ..."""
import gzip, json, sys
F = json.load(gzip.open('models/grading/history/MLB_FINALS.json.gz', 'rt'))
worst = 0; n = 0
print(f"{'game':28} {'live app (Oct)':>22} {'desk runner (now)':>22}  diff")
for p in sys.argv[1:]:
    d = json.load(open(p))
    for g in d['games']:
        f = F.get(str(g['gamePk'])); pre = ((f or {}).get('pregame') or {}).get('models', {}).get('eff')
        e = (g.get('models') or {}).get('eff')
        if not (pre and e): continue
        diffs = [abs(e[k] - pre[k]) for k in ('runsAway', 'runsHome', 'homeWinProb')]
        worst = max(worst, max(diffs)); n += 1
        print(f"{g['date']} {g['away'][:10]:>10}@{g['home'][:10]:10} {pre['runsAway']:6.3f} {pre['runsHome']:6.3f} {pre['homeWinProb']:.4f}   "
              f"{e['runsAway']:6.3f} {e['runsHome']:6.3f} {e['homeWinProb']:.4f}  {max(diffs):.4f}")
print(f'{n} games compared; largest difference {worst:.4f}')
