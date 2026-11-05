"""Walk-forward ratings on the Kaggle alimoh89 file (tennisexplorer results + bet365 open/close ML, handicap, totals, 2014-25,
ATP tour/Challenger, WTA tour/ITF). Replicates the live tennis model's core inputs (Elo, surface Elo, games-won rating, n,
scoring style = games per set + tiebreak-set rate) so Model A/B can be tested where closing totals and handicaps exist.
Usage: prep.py KAGGLE_CSV OUT_PARQUET"""
import sys, re, numpy as np, pandas as pd
from collections import defaultdict
src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, low_memory=False)
d = d[~d.comment.isin(['not_played', 'invalid_score'])].copy()
d['t'] = pd.to_datetime(d.match_date_formatted, errors='coerce'); d = d[d.t.notna()].sort_values('t', kind='stable').reset_index(drop=True)
d['sex'] = np.where(d.tour == 'ATP', 'm', 'w')
tl = d.tournament.str.lower()
d['level'] = np.where(tl.str.contains('challenger'), 'Challenger', np.where(d.tournament.str.contains('ITF'), 'ITF', 'Tour'))
d['bo'] = d.match_format.astype(int)
d['surf'] = d.surface.replace({'indoors': 'hard'})   # indoor hard; tennisexplorer 'indoors' is mostly hard
d['ret'] = d.comment != 'completed'
sets = [(f'set{i}_score_a', f'set{i}_score_b') for i in range(1, 6)]
# randomise orientation (player_a is usually the winner)
rng = np.random.default_rng(7); flip = rng.random(len(d)) < .5
def sw(a, b):
    A, B = d[a].copy(), d[b].copy(); d.loc[flip, a] = B[flip]; d.loc[flip, b] = A[flip]
for a, b in [('player_a_url', 'player_b_url'), ('player_a', 'player_b'), ('total_games_a', 'total_games_b'), ('total_sets_a', 'total_sets_b')] + sets + \
        [(f'{bk}_odds_a{s}', f'{bk}_odds_b{s}') for bk in ('bet365', 'Betfair', 'Ladbrokes', 'Unibet') for s in ('', '_opening')]:   # handicap columns stay favourite-first (not flipped)
    sw(a, b)
# winner: completed -> more sets; retired -> player_a in the raw file (winner listed first) = not flipped
d['y'] = np.where(d.ret, (~flip).astype(float), (d.total_sets_a > d.total_sets_b).astype(float))
E = defaultdict(lambda: 1500.); ES = defaultdict(lambda: 1500.); N = defaultdict(int); G = defaultdict(float); NG = defaultdict(int)
ST = defaultdict(lambda: [0., 0., 0.])   # sets, games in those sets, tiebreak sets (decayed)
LAST = {}
cols = defaultdict(list)
for r in d.itertuples(index=False):
    a, b = r.player_a_url, r.player_b_url; s = r.surf; sx = r.sex
    ka, kb = (sx, a), (sx, b)
    cols['elo_a'].append(E[ka]); cols['elo_b'].append(E[kb]); cols['elos_a'].append(ES[(ka, s)]); cols['elos_b'].append(ES[(kb, s)])
    cols['n_a'].append(N[ka]); cols['n_b'].append(N[kb]); cols['g_a'].append(G[ka]); cols['g_b'].append(G[kb])
    for p, k in (('a', ka), ('b', kb)):
        st = ST[k]; cols[f'sets_{p}'].append(st[0]); cols[f'gps_{p}'].append(st[1] / st[0] if st[0] else np.nan); cols[f'tb_{p}'].append(st[2] / st[0] if st[0] else np.nan)
        cols[f'lay_{p}'].append((r.t - LAST[k]).days if k in LAST else 999)
    # updates
    ea = 1 / (1 + 10 ** ((E[kb] - E[ka]) / 400)); y = r.y
    K = lambda n: 250 / (n + 5) ** 0.4
    E[ka] += K(N[ka]) * (y - ea); E[kb] -= K(N[kb]) * (y - ea)
    es = 1 / (1 + 10 ** ((ES[(kb, s)] - ES[(ka, s)]) / 400)); ES[(ka, s)] += K(N[ka]) * (y - es); ES[(kb, s)] -= K(N[kb]) * (y - es)
    N[ka] += 1; N[kb] += 1; LAST[ka] = LAST[kb] = r.t
    sc = [(getattr(r, x), getattr(r, z)) for x, z in sets]
    sc = [(x, z) for x, z in sc if 6 <= max(x, z) <= 7 and (abs(x - z) >= 2 or max(x, z) == 7)]
    if sc and not r.ret:
        tga = sum(x for x, _ in sc); tgb = sum(z for _, z in sc)
        sa = tga / (tga + tgb); e = 1 / (1 + np.exp(-(G[ka] - G[kb])))
        G[ka] += 0.9 / (min(NG[ka], 60) + 8) ** .5 * (sa - e) * 4; G[kb] -= 0.9 / (min(NG[kb], 60) + 8) ** .5 * (sa - e) * 4
        NG[ka] += 1; NG[kb] += 1
        for k in (ka, kb):
            st = ST[k]; dec = 0.985 ** len(sc)
            st[0] = st[0] * dec + len(sc); st[1] = st[1] * dec + sum(x + z for x, z in sc); st[2] = st[2] * dec + sum(x + z == 13 for x, z in sc)
F = pd.concat([d.reset_index(drop=True), pd.DataFrame(cols)], axis=1)
keep = ['t', 'sex', 'level', 'bo', 'surf', 'tournament', 'round', 'ret', 'y', 'player_a', 'player_b', 'player_a_url', 'player_b_url',
        'total_games_a', 'total_games_b', 'total_games', 'total_sets_a', 'total_sets_b',
        'bet365_odds_a', 'bet365_odds_b', 'bet365_odds_a_opening', 'bet365_odds_b_opening',
        'bet365_handicap', 'bet365_odds_handicap_a', 'bet365_odds_handicap_b', 'bet365_odds_handicap_a_opening', 'bet365_odds_handicap_b_opening',
        'bet365_total_games_line', 'bet365_odds_over', 'bet365_odds_under', 'bet365_odds_over_opening', 'bet365_odds_under_opening',
        'Betfair_odds_a', 'Betfair_odds_b'] + list(cols)
F[keep].to_parquet(out)
print(len(F), 'rows ->', out)
