"""Parlay true-odds pricer (2026-10-17): joint hit chance of any set of legs from our own models + measured correlations.

python3 pricer.py NFL "QB_pass_yds:over:0.56" "WR1_rec_yds:over:0.54" "OPP_QB_pass_yds:over:0.52" "G_over:over:0.51" --book +1100
python3 pricer.py CFB "FAV_COVER:0.52" "OVER:0.50" --ratio 0.45 --book +240
python3 pricer.py SOCCER "F_win:over:0.62" "over25:over:0.55" "fav_cs:under:0.6" --ratio 0.62 --book +300   # fav win + over 2.5 + opponent scores
python3 pricer.py ANY "0.55" "0.57" "0.54" --book +600            # legs from different games (independent)

Each leg = ROLE_STAT:over|under:P, where P is OUR model's chance that leg hits at the posted line (Bet Desk shows it).
Roles are by projection rank on the team (NFL QB, RB1, WR1/WR2 = top two pass catchers by projected yards, REC3; NBA P1-P5 by
projected points; NHL S1-S6 by projected points, GOALIE; MLB B1-B9 batting slot, SP = that team's starter; G_ = team/game
legs: G_cover, G_team_total, G_over, G_win). OPP_ = the other team. Stats are in parlay_corr.json (app/data).
Method: Gaussian copula on latent (tetrachoric) correlations measured between our projection residuals. Validated on
held-out 2022-25 NFL stacks of 3-5 legs: mean error vs actual hit rate under 1%, spread about 3% (copula_test.py).
Prints fair joint %, fair American odds, the no-correlation price, the correlation multiplier, and EV / quarter-Kelly at
the book's price. Rule: bet only if EV >= +5% (parlay prices carry more model error than singles)."""
import argparse, json, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copula import joint_gauss
here = os.path.dirname(os.path.abspath(__file__))
TAB = json.load(open(os.path.join(here, '../../app/data/parlay_corr.json')))
am2dec = lambda a: 1 + (a / 100 if a > 0 else 100 / -a)
def dec2am(d): return f'+{round(100*(d-1))}' if d >= 2 else f'{round(-100/(d-1))}'

def price(sport, legs, book=None, ratio=None):
    """legs: list of (name, direction, p). Returns dict."""
    p = np.array([l[2] for l in legs], float); k = len(legs); R = np.eye(k); src = []
    if sport in TAB and sport not in ('game_spread_total', 'note', 'SOCCER'):
        T = TAB[sport]; idx = {n: i for i, n in enumerate(T['legs'])}
        for a in range(k):
            for b in range(a + 1, k):
                na, nb = legs[a][0], legs[b][0]
                if na in idx and nb in idx:
                    r = T['R'][idx[na]][idx[nb]] * (1 if legs[a][1] == legs[b][1] else -1)
                else: r = 0.0; src.append(f'no table entry for {na} x {nb}: treated as independent')
                R[a, b] = R[b, a] = r
    elif sport == 'SOCCER':
        T = TAB['SOCCER']; fp = ratio if ratio is not None else 0.5; band = next(b for b in T['by_fav_prob'] if b['lo'] <= fp < b['hi'])
        idx = {n: i for i, n in enumerate(T['legs'])}
        for a in range(k):
            for b in range(a + 1, k):
                r = band['R'][idx[legs[a][0]]][idx[legs[b][0]]] * (1 if legs[a][1] == legs[b][1] else -1); R[a, b] = R[b, a] = r
    elif sport in ('CFB', 'NFL', 'CBB', 'NBA') and ratio is not None:
        rows = TAB['game_spread_total'][sport]; rho = next((r['rho'] for r in rows if r['lo'] <= ratio < r['hi']), rows[-1]['rho'])
        names = [l[0] for l in legs]; i, j = names.index('FAV_COVER'), names.index('OVER')
        R[i, j] = R[j, i] = rho * (1 if legs[i][1] == legs[j][1] else -1)
    pj = joint_gauss(p, R); ind = float(np.prod(p))
    out = dict(legs=k, fair_joint=round(pj, 4), fair_odds=dec2am(1 / pj), no_corr_odds=dec2am(1 / ind), corr_multiplier=round(pj / ind, 3))
    if book is not None:
        d = am2dec(book); ev = pj * d - 1; kelly = max(0.0, (pj * d - 1) / (d - 1)) / 4
        out.update(book_odds=book, ev_pct=round(100 * ev, 1), quarter_kelly_pct_bankroll=round(100 * kelly, 2),
                   verdict='BET' if ev >= 0.05 else ('thin' if ev > 0 else 'pass'))
    if src: out['warnings'] = src
    return out

def parse(s, sport):
    parts = s.split(':')
    if sport == 'ANY': return ('leg', 'over', float(parts[-1]))
    if sport in ('CFB', 'CBB') or parts[0] in ('FAV_COVER', 'OVER', 'DOG_COVER', 'UNDER'):
        nm = {'DOG_COVER': ('FAV_COVER', 'under'), 'UNDER': ('OVER', 'under')}.get(parts[0], (parts[0], 'over'))
        return (nm[0], nm[1], float(parts[-1]))
    return (parts[0], parts[1].lower(), float(parts[2]))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('sport'); ap.add_argument('legs', nargs='+')
    ap.add_argument('--book', type=float); ap.add_argument('--ratio', type=float, help='|spread| / total for FAV_COVER + OVER legs; for SOCCER the favourite no-vig win chance')
    a = ap.parse_args(); sp = a.sport.upper()
    for k_, v in price(sp, [parse(s, sp) for s in a.legs], a.book, a.ratio).items(): print(f'{k_:>26}: {v}')
