"""NBA-1 injury-news repricer (2026-10-11): when news breaks (a player ruled out, upgraded, or questionable), reprice the
GAME and every TEAMMATE / OPPONENT PROP in one step, before and after the news, against the lines on the board now.

usage (run from models/nba-team):
  python3 reprice.py HOME AWAY --date 2026-11-02 --line -4.5 --total 226.5 --news "SA:Victor Wembanyama:out"
         [--news "NY:Jalen Brunson:0.4"]  [--out-home "A;B"] [--out-away "C"]   (already-known absences, priced in both)
         [--props props.csv]   (name,stat,line: prop lines currently posted; stat in pts reb ast fg3m stl blk tov pra)
--news TEAM:Name:out  the player will not play; TEAM:Name:in  he will play (was listed out); TEAM:Name:0.4  questionable,
with your chance he plays. --line is the home spread in betting notation (home -4.5 = favoured by 4.5).

Why: the team model's opener edge came from knowing who sat (CLV +1.18 pts at 3+ point gaps on games with news after
the open, vs +0.35 without). Books move sides first and teammates' props later, so the first minutes after news are
where the edge is. Uses ../nba-player/project.py (minutes-first player model, NBA-2) for minutes and props, and
predict.py (lineup team model) with those minutes for the game. Prints the before/after fair line, how far the line
should move, the cover/over chance at the CURRENT line (opener weights: the line has not caught up yet), and the props
whose fair value moved most."""
import argparse, os, re, subprocess, sys, tempfile, numpy as np, pandas as pd
here = os.path.dirname(os.path.abspath(__file__)); PM = os.path.normpath(f'{here}/../nba-player')
ap = argparse.ArgumentParser(); ap.add_argument('home'); ap.add_argument('away')
ap.add_argument('--date', required=True); ap.add_argument('--line', type=float, required=True); ap.add_argument('--total', type=float, required=True)
ap.add_argument('--news', action='append', default=[]); ap.add_argument('--out-home', default=''); ap.add_argument('--out-away', default='')
ap.add_argument('--props'); ap.add_argument('--neutral', action='store_true')
A = ap.parse_args()
H, AW = A.home.upper(), A.away.upper()

def scen(after):
    """out lists and questionable lists per team for the before / after state."""
    out = {H: [x for x in A.out_home.split(';') if x.strip()], AW: [x for x in A.out_away.split(';') if x.strip()]}; q = {H: [], AW: []}
    for n in A.news:
        t, name, what = [x.strip() for x in n.split(':')]; t = t.upper()
        if t not in out: raise SystemExit(f'--news team {t} is not {H} or {AW}')
        w = what.lower()
        if w == 'in':        # was ruled out before the news, now plays
            if not after: out[t].append(name)
        elif after:
            if w == 'out': out[t].append(name)
            else: q[t].append(f'{name}:{float(w)}')
    return out, q

def run_players(after, tmp):
    out, q = scen(after)
    sp = -A.line   # home expected margin
    g = pd.DataFrame([dict(date=A.date, team=H, opp=AW, home=1, spread=sp, total=A.total, out=';'.join(out[H]), add='', q=';'.join(q[H])),
                      dict(date=A.date, team=AW, opp=H, home=0, spread=-sp, total=A.total, out=';'.join(out[AW]), add='', q=';'.join(q[AW]))])
    gp = f'{tmp}/games_{after}.csv'; g.to_csv(gp, index=False); op = f'{tmp}/proj_{after}.csv'
    args = [sys.executable, f'{PM}/project.py', PM, gp, A.props or '-', op]
    r = subprocess.run(args, capture_output=True, text=True, cwd=PM)
    if r.returncode: raise SystemExit(r.stderr[-2000:])
    P = pd.read_csv(op); L = pd.read_csv(op.replace('.csv', '_lines.csv')) if A.props and os.path.exists(op.replace('.csv', '_lines.csv')) else None
    return P, L, out, q

def run_game(after, P, out, q, tmp):
    mp = f'{tmp}/mins_{after}.csv'; P[['name', 'min_proj']].to_csv(mp, index=False)
    args = [sys.executable, 'predict.py', H, AW, '--line', str(A.line), '--total', str(A.total), '--at', 'open', '--minutes', mp,
            '--out-home', ';'.join(out[H]), '--out-away', ';'.join(out[AW]), '--q-home', ';'.join(q[H]), '--q-away', ';'.join(q[AW])]
    if A.neutral: args.append('--neutral')
    r = subprocess.run(args, capture_output=True, text=True, cwd=here)
    if r.returncode: raise SystemExit(r.stderr[-2000:])
    lines = r.stdout.splitlines()
    hdr = next(i for i, l in enumerate(lines) if 'margin (home)' in l)
    row = lines[hdr + 1]
    lab = re.match(r'\s*([a-z ]+?)\s+-?\d', row).group(1)
    vals = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', row[row.index(lab) + len(lab):])]
    keys = ['home_pts', 'away_pts', 'margin', 'total', 'p_home_win', 'p_cover', 'p_win_mkt', 'p_over'][:len(vals)]
    return dict(zip(keys, vals))

with tempfile.TemporaryDirectory() as tmp:
    P0, L0, out0, q0 = run_players(False, tmp); P1, L1, out1, q1 = run_players(True, tmp)
    G0 = run_game(False, P0, out0, q0, tmp); G1 = run_game(True, P1, out1, q1, tmp)

print(f'{H} vs {AW} {A.date} | news: ' + '; '.join(A.news))
fair0, fair1 = -G0['margin'], -G1['margin']
print(f"\nGAME          before news   after news   change")
print(f"fair spread ({H})  {fair0:+8.1f}    {fair1:+8.1f}    {fair1 - fair0:+.1f}   (board {A.line:+.1f})")
print(f"fair total         {G0['total']:8.1f}    {G1['total']:8.1f}    {G1['total'] - G0['total']:+.1f}   (board {A.total:.1f})")
if 'p_cover' in G1: print(f"P({H} covers {A.line:+.1f}) {G0['p_cover']:.3f} -> {G1['p_cover']:.3f}   P(over {A.total}) {G0.get('p_over', float('nan')):.3f} -> {G1.get('p_over', float('nan')):.3f}")
mv = fair1 - fair0
print(f"Expected move: the news is worth {abs(mv):.1f} points of spread toward {AW if mv > 0 else H}." if abs(mv) >= 0.5 else 'The news is worth under half a point of spread.')

M = P0.merge(P1, on=['name', 'team'], how='outer', suffixes=('_before', '_after'))
for s in ['min_proj', 'pts', 'reb', 'ast', 'pra', 'fg3m']: M[f'd_{s}'] = M[f'{s}_after'] - M[f'{s}_before']
M = M[M.min_proj_after.notna()]; M = M.reindex(M.d_pra.abs().sort_values(ascending=False).index)
print('\nPLAYERS whose projection moved most (after-news mean, change)')
show = M.head(12)
for _, r in show.iterrows():
    print(f"  {r['name']:<24s} {r.team:<5s} min {r.min_proj_after:5.1f} ({r.d_min_proj:+.1f})  pts {r.pts_after:5.1f} ({r.d_pts:+.1f})  "
          f"reb {r.reb_after:4.1f} ({r.d_reb:+.1f})  ast {r.ast_after:4.1f} ({r.d_ast:+.1f})  pra {r.pra_after:5.1f} ({r.d_pra:+.1f})")
if L1 is not None:
    L = L1.merge(L0[['name', 'stat', 'line', 'p_over']], on=['name', 'stat', 'line'], how='left', suffixes=('', '_before'))
    L['move'] = L.p_over - L.p_over_before
    L = L.reindex(L.move.abs().sort_values(ascending=False).index)
    print('\nPOSTED PROPS at the current lines (lean = model edge after the news; overs need 58%, unders 55%)')
    print(L[['name', 'stat', 'line', 'mean', 'p_over_before', 'p_over', 'move', 'lean']].fillna({'lean': ''}).round(3).to_string(index=False))
print('\nAct before the line moves: the backtest edge is in the minutes after news, not after the market has repriced.')
