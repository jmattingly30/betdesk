"""Price one NBA game: projected score, margin, total, win probability, cover / over probability, with any players out.

usage: python3 predict.py HOME AWAY [--out-home "A;B"] [--out-away "C"] [--add-home "X"] [--add-away "Y"]
                          [--line -4.5] [--total 228.5] [--at open|close] [--minutes projections.csv] [--neutral]
Team codes are ESPN's (SA, NY, GS, LAL, UTAH, WSH, NO, ...). --line is the HOME spread in betting notation (home -4.5 =
favoured by 4.5). --out-* removes players from the team's latest roster; --add-* adds players who changed teams.
--minutes takes the output of ../nba-player/project.py (columns name, min_proj) so the player model's minutes layer
decides who plays how much; otherwise minutes are each player's recent average, scaled to 240.
The "full strength" row prices the same game with nobody out, so the difference is what the absences are worth.

2026-10-09 audit:
  --q-home "Name:0.6;Name2:0.25" / --q-away: questionable players with your chance each plays (N1). Every combination
      of who plays is priced and weighted, so margin, total and every probability are the mixture, not a guess.
  --playoff adds the fitted playoff home edge (N4).
  With --line, P(home win) is market-anchored: centred on line + b x (model - line), sigma growing with the total (N3)."""
import sys, json, argparse, re, itertools, unicodedata, numpy as np, pandas as pd
from scipy.stats import norm
STATS = ['pts', 'fga', 'fta', 'fg3a', 'fg3m', 'ast', 'tov', 'oreb', 'dreb', 'stl', 'blk', 'pf']   # same as lineup.py
RC = ['r_' + s for s in STATS]
ap = argparse.ArgumentParser(); ap.add_argument('home'); ap.add_argument('away')
for a in ['--out-home', '--out-away', '--add-home', '--add-away', '--minutes', '--q-home', '--q-away']: ap.add_argument(a, default='')
ap.add_argument('--playoff', action='store_true')
ap.add_argument('--line', type=float); ap.add_argument('--total', type=float); ap.add_argument('--at', default='open'); ap.add_argument('--neutral', action='store_true')
A = ap.parse_args()
def norm_(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', re.sub(r'[^a-z ]', '', s)).split())
cfg = json.load(open('model_config.json')); season = cfg['season']; k = cfg['k']; W = np.array(cfg['lineup_weights'][str(season)])
RC = cfg.get('rc', RC)
wo, wd = W[1:1 + len(RC)], W[1 + len(RC):1 + 2 * len(RC)]
tr = pd.read_csv('team_residual_ratings.csv').set_index('team')
cp = pd.read_parquet('data/current_players.parquet'); cp['key'] = cp.name.map(norm_)
mins = None
if A.minutes:
    mm = pd.read_csv(A.minutes); mins = dict(zip(mm.name.map(norm_), mm.min_proj))
# centre of the lineup term = the average lineup matchup in the rating season
la = pd.read_parquet('data/lineup_agg.parquet'); gm = pd.read_parquet('data/games.parquet')[['game_id', 'season', 'home', 'away']]
la = la.merge(gm, on='game_id'); la = la[la.season == season]
la['opp'] = np.where(la.team == la.home, la.away, la.home)
oth = la.set_index(['game_id', 'team'])[RC]
Y = oth.reindex(pd.MultiIndex.from_arrays([la.game_id, la.opp])).to_numpy()
centre = float(np.nanmean(la[RC].to_numpy() @ wo + Y @ wd))

def lineup(team, out, add, use_minutes=True):
    out = {norm_(x) for x in out.split(';') if x.strip()}; add = {norm_(x) for x in add.split(';') if x.strip()}
    r = cp[((cp.team == team) & (cp.season_last == season) & ~cp.key.isin(out)) | cp.key.isin(add)].copy()
    unk = out - set(cp[cp.team == team].key)
    if unk: print(f'{team}: not on the latest roster (ignored):', sorted(unk))
    if mins is not None and use_minutes: r['min_proj'] = r.key.map(mins).fillna(0)
    r = r[r.min_proj > 0].sort_values('min_proj', ascending=False)
    # a ten-man rotation: deep bench players who rarely play take minutes only if the rotation is short
    r = r.head(max(10, (r.min_proj >= 15).sum()))
    r['min'] = r.min_proj / r.min_proj.sum() * 240
    return r, (r['min'].to_numpy() / 48) @ r[RC].to_numpy()

def lus(xh, xa): return xh @ wo + xa @ wd - centre, xa @ wo + xh @ wd - centre
def score(xh, xa, full=None):
    lu_h, lu_a = lus(xh, xa)
    H, Aw = tr.loc[A.home], tr.loc[A.away]; hca = 0 if A.neutral else H.hca
    poss = H.tempo + Aw.tempo - H.T0
    ph = H.mu + H.O - Aw.D + hca + k * lu_h; pa = H.mu + Aw.O - H.D - hca + k * lu_a
    sh, sa = ph * poss / 100, pa * poss / 100
    if full is not None:   # extra weight on absences (fit in the backtest): the market values them more than the shrunk lineup term
        fh, fa = lus(*full); extra = cfg['absence_boost'] * ((lu_h - fh) - (lu_a - fa)) * poss / 100
        sh, sa = sh + extra / 2, sa - extra / 2
    if A.playoff: sh, sa = sh + cfg.get('playoff_home_edge', 0) / 2, sa - cfg.get('playoff_home_edge', 0) / 2
    return sh, sa

def qparse(x):
    out = []
    for part in x.split(';'):
        if ':' in part: n_, p_ = part.rsplit(':', 1); out.append((n_.strip(), min(1.0, max(0.0, float(p_)))))
    return out
QH, QA = qparse(A.q_home), qparse(A.q_away)
if len(QH) + len(QA) > 6: raise SystemExit('at most 6 questionable players')

def price(xh, xa, label, full=None, scen=None):
    """scen: list of (probability, xh, xa) lineups; when given, every output is the probability-weighted mixture."""
    scen = scen or [(1.0, xh, xa)]
    S = [(p_,) + score(a_, b_, full) for p_, a_, b_ in scen]
    P = np.array([x[0] for x in S]); SH = np.array([x[1] for x in S]); SA = np.array([x[2] for x in S])
    sh, sa = float(P @ SH), float(P @ SA)
    m, t = sh - sa, sh + sa
    MS, TS = SH - SA, SH + SA
    row = {'': label, A.home: round(sh, 1), A.away: round(sa, 1), 'margin (home)': round(m, 1), 'total': round(t, 1),
           'P(home win)': round(float(P @ norm.cdf(MS / cfg['sigma_margin'])), 3)}
    if A.line is not None:
        b = cfg['blend_open'] if A.at == 'open' else cfg['blend_close']
        bs, sd = (b['b_spread_open'], b['sd_ats_open']) if A.at == 'open' else (b['b_spread'], b['sd_ats'])
        row['P(home covers %+.1f)' % A.line] = round(float(P @ norm.cdf(bs * (MS + A.line) / sd)), 3)
        wm = cfg.get('win_market')
        if wm:   # market-anchored win probability (N3)
            sg = wm['s0'] * np.sqrt((A.total if A.total is not None else t) / 220)
            row['P(home win), market-anchored'] = round(float(P @ norm.cdf((-A.line + wm['b'] * (MS + A.line)) / sg)), 3)
    if A.total is not None:
        b = cfg['blend_open'] if A.at == 'open' else cfg['blend_close']
        bt, st = (b['b_total_open'], b['sd_total_open']) if A.at == 'open' else (b['b_total'], b['sd_total'])
        row['P(over %.1f)' % A.total] = round(float(P @ norm.cdf(bt * (TS - A.total) / st)), 3)
    return row
rh, xh = lineup(A.home, A.out_home, A.add_home); ra, xa = lineup(A.away, A.out_away, A.add_away)
_, fh = lineup(A.home, '', A.add_home, False); _, fa = lineup(A.away, '', A.add_away, False)   # full strength: recent minutes
scen = None
if QH or QA:
    scen = []
    Q = [('h', n_, p_) for n_, p_ in QH] + [('a', n_, p_) for n_, p_ in QA]
    for plays in itertools.product([True, False], repeat=len(Q)):
        pr_ = float(np.prod([p_ if pl else 1 - p_ for (_, _, p_), pl in zip(Q, plays)]))
        if pr_ == 0: continue
        oh = ';'.join([A.out_home] + [n_ for (sd_, n_, _), pl in zip(Q, plays) if sd_ == 'h' and not pl])
        oa = ';'.join([A.out_away] + [n_ for (sd_, n_, _), pl in zip(Q, plays) if sd_ == 'a' and not pl])
        scen.append((pr_, lineup(A.home, oh, A.add_home)[1], lineup(A.away, oa, A.add_away)[1]))
    print('questionable:', ', '.join(f'{n_} {p_:.0%}' for _, n_, p_ in Q), f'-> {len(scen)} lineup combinations priced')
rows = [price(xh, xa, 'with absences' if (A.out_home or A.out_away or scen) else 'projection', full=(fh, fa), scen=scen)]
if A.out_home or A.out_away: rows.append(price(fh, fa, 'full strength', full=(fh, fa)))
pd.set_option('display.width', 200)
print(pd.DataFrame(rows).to_string(index=False))
for t, r in [(A.home, rh), (A.away, ra)]:
    print(f'\n{t} expected rotation:', ', '.join(f'{n} {m:.0f}' for n, m in zip(r.name, r['min'])))
print('\nCover / over probabilities are market-anchored: only part of the model-minus-line gap is treated as real '
      '(%s weights from the backtest).' % A.at)
