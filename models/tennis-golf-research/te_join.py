"""Join the tennisexplorer odds set (Kaggle alimoh89: 300k matches 2014-25, bet365/Betfair/Ladbrokes/Unibet open+close,
ML + game handicap + total games, set scores) to our walk-forward model rows (deuce, to 2024-12).
Output ../data/te_{atp,wta}.parquet oriented to our player a."""
import sys
import numpy as np, pandas as pd
from bridge import norm

TE = pd.read_csv('../../kg/alimoh89_tennis-results-and-betting-odds-20142025/tennis_matches_2014_2025.csv', low_memory=False)
TE['date'] = pd.to_datetime(TE.match_date_formatted, errors='coerce').dt.normalize()
TE['na'] = TE.player_a.map(norm); TE['nb'] = TE.player_b.map(norm)
TE['key'] = [' | '.join(sorted((x, y))) for x, y in zip(TE.na, TE.nb)]
BK = ['bet365', 'Betfair', 'Ladbrokes', 'Unibet']
for tour, tcode in (('atp', 'ATP'), ('wta', 'WTA')):
    F = pd.read_parquet(f'../data/feat_{tour}.parquet').join(pd.read_parquet(f'../data/pred_{tour}.parquet'))
    F = F[F.date >= '2013-12-01']
    M = pd.read_parquet(f'../data/{tour}_matches.parquet', columns=['winner_id', 'winner_name', 'loser_id', 'loser_name'])
    names = dict(zip(M.winner_id, M.winner_name)); names.update(dict(zip(M.loser_id, M.loser_name)))
    F['na_'] = F.a.map(names).map(norm); F['nb_'] = F.b.map(names).map(norm)
    F['key'] = [' | '.join(sorted((x, y))) for x, y in zip(F.na_, F.nb_)]
    F['fidx'] = F.index
    T = TE[TE.tour == tcode]
    J = T.merge(F[['key', 'date', 'fidx', 'na_']], on='key', suffixes=('', '_f'))
    J['dd'] = (J.date - J.date_f).dt.days
    J = J[(J.dd >= -3) & (J.dd <= 21)].sort_values('dd').drop_duplicates('match_link').drop_duplicates('fidx')
    flip = J.na != J.na_  # te player_a is our b
    out = pd.DataFrame({'fidx': J.fidx.values, 'te_date': J.date.values, 'te_tour': J.tournament.values, 'flip': flip.values,
                        'completed': J.comment.values, 'total_games': J.total_games.values})
    ga = np.where(flip, J.total_games_b, J.total_games_a); gb = np.where(flip, J.total_games_a, J.total_games_b)
    out['marA'] = ga - gb
    sw = lambda x, y: np.where(flip, y, x)
    out['set1_a'] = (sw(J.set1_score_a, J.set1_score_b) > sw(J.set1_score_b, J.set1_score_a)).astype(float)
    for bk in BK:
        for sfx in ('', '_opening'):
            a, b = J.get(f'{bk}_odds_a{sfx}'), J.get(f'{bk}_odds_b{sfx}')
            out[f'{bk}_a{sfx}'] = sw(a, b); out[f'{bk}_b{sfx}'] = sw(b, a)
            if bk != 'Betfair':
                # handicap columns are favourite-first (checked on settled lines): favourite gives |H| games at price 'a'
                fav_te_a = (J[f'{bk}_odds_a'] < J[f'{bk}_odds_b']).values
                fav_our_a = fav_te_a ^ flip.values
                ha, hb = J.get(f'{bk}_odds_handicap_a{sfx}').values, J.get(f'{bk}_odds_handicap_b{sfx}').values
                out[f'{bk}_ha{sfx}'] = np.where(fav_our_a, ha, hb); out[f'{bk}_hb{sfx}'] = np.where(fav_our_a, hb, ha)
                out[f'{bk}_over{sfx}'] = J.get(f'{bk}_odds_over{sfx}').values; out[f'{bk}_under{sfx}'] = J.get(f'{bk}_odds_under{sfx}').values
        if bk != 'Betfair':
            out[f'{bk}_hcp'] = np.where(fav_our_a, -J[f'{bk}_handicap'].abs(), J[f'{bk}_handicap'].abs())
            out[f'{bk}_tot'] = J[f'{bk}_total_games_line'].values
    out = out.set_index('fidx').join(F.drop(columns=['na_', 'nb_', 'key', 'fidx']))
    out.to_parquet(f'../data/te_{tour}.parquet')
    print(tour, 'te rows', len(T), 'joined', len(out), 'share of model rows 2014-24 matched',
          round(len(out) / len(F[F.date >= '2014-01-01']), 3), '| y agrees with te result:',
          round(((out.marA > 0) == (out.y == 1))[out.completed == 'completed'].mean(), 4))
    print(out.groupby('level').size().to_dict())
