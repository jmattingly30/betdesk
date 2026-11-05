"""Join Grand Slam odds (tennis-data.co.uk via deuce) to engine features."""
import unicodedata, re
import numpy as np, pandas as pd

SLAM = {'Australian Open': 'AO', 'French Open': 'RG', 'Roland Garros': 'RG', 'Wimbledon': 'W', 'US Open': 'USO',
        'Us Open': 'USO'}


def norm(x):
    x = unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z]', '', x)


def td_surname(x):
    # "Moutet C." -> "moutet"; "Del Potro J.M." -> "delpotro"; "Auger-Aliassime F." -> "augeraliassime"
    x = str(x).strip()
    parts = x.split(' ')
    if len(parts) > 1 and re.match(r'^([A-Z]\.?-?)+$', parts[-1].replace('.', '').replace('-', '') or 'x'):
        parts = parts[:-1]
    return norm(' '.join(parts))


def join(F, tour):
    O = pd.read_parquet(f'../data/{tour}_odds.parquet')
    O['date'] = pd.to_datetime(O.Date)
    O['slam'] = O.Tournament.map(SLAM)
    O['yr'] = O.date.dt.year
    O['ws'] = O.Winner.map(td_surname); O['ls'] = O.Loser.map(td_surname)
    G = F[F.tname.isin(['Australian Open', 'Roland Garros', 'Wimbledon', 'US Open', 'Us Open'])].copy()
    G['slam'] = G.tname.map(SLAM); G['yr'] = G.date.dt.year
    # Australian Open sometimes listed with start date late Dec
    G.loc[G.date.dt.month == 12, 'yr'] += 1
    G = G[~G.rnd.astype(str).str.startswith('Q')]
    out = []
    Gg = {k: v for k, v in G.groupby(['slam', 'yr'])}
    M = pd.read_parquet(f'../data/{tour}_matches.parquet', columns=['winner_id', 'winner_name', 'loser_id', 'loser_name'])
    pn = pd.concat([M[['winner_id', 'winner_name']].set_axis(['id', 'nm'], axis=1), M[['loser_id', 'loser_name']].set_axis(['id', 'nm'], axis=1)]).drop_duplicates('id').set_index('id').nm.to_dict()
    for (sl, yr), o in O.groupby(['slam', 'yr']):
        g = Gg.get((sl, yr))
        if g is None:
            continue
        an = g.a.map(pn).map(norm).values; bn = g.b.map(pn).map(norm).values
        for row in o.itertuples():
            hit = None
            for i in range(len(g)):
                if (an[i].endswith(row.ws) and bn[i].endswith(row.ls)) or (an[i].endswith(row.ls) and bn[i].endswith(row.ws)):
                    hit = i
                    break
            if hit is None:
                continue
            gi = g.iloc[hit]
            a_won = gi.y == 1
            rec = dict(idx=g.index[hit], PSW=row.PSW, PSL=row.PSL, B365W=row.B365W, B365L=row.B365L,
                       MaxW=row.MaxW, MaxL=row.MaxL, AvgW=row.AvgW, AvgL=row.AvgL, Comment=row.Comment)
            out.append(rec)
    J = pd.DataFrame(out).drop_duplicates('idx').set_index('idx')
    return J


