"""Surface for 2026 OddsPapi events: match the host city to past events (deuce 2017-24, TML 2025) by city + month."""
import re, unicodedata
import pandas as pd
from collections import Counter, defaultdict


def city(s):
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'^(atp challenger|atp|wta 125k|wta|challenger|itf)\s+', '', s)
    s = s.split(',')[0]
    s = re.sub(r'\b(men|women|singles|ch|m\d+|w\d+|\d+)\b', ' ', s)
    s = re.sub(r'[^a-z ]', ' ', s)
    return ' '.join(s.split())


def build():
    rows = []
    for t in ('atp', 'wta'):
        d = pd.read_parquet(f'../data/{t}_matches.parquet', columns=['tourney_name', 'surface', 'tourney_start_date'])
        d = d[d.tourney_start_date >= '2017'].drop_duplicates(['tourney_name', 'tourney_start_date'])
        rows.append(pd.DataFrame(dict(c=d.tourney_name.map(city), s=d.surface, m=pd.to_datetime(d.tourney_start_date).dt.month)))
    t = pd.read_csv('../../dl/Tennismylife_TML-Database_master_2025.csv').drop_duplicates('tourney_id')
    rows.append(pd.DataFrame(dict(c=t.tourney_name.map(city), s=t.surface, m=pd.to_datetime(t.tourney_date.astype(str)).dt.month)))
    R = pd.concat(rows)
    R = R[R.s.isin(['Hard', 'Clay', 'Grass', 'Carpet'])]
    by_cm = defaultdict(Counter); by_c = defaultdict(Counter)
    for r in R.itertuples():
        by_cm[(r.c, r.m)][r.s] += 1; by_c[r.c][r.s] += 1
    return by_cm, by_c


BY_CM, BY_C = build()
SLAM = {'australian open': 'Hard', 'french open': 'Clay', 'wimbledon': 'Grass', 'us open': 'Hard'}


def surface(name, month):
    n = str(name).lower()
    for k, v in SLAM.items():
        if n.startswith(k): return v
    c = city(name)
    for mm in (month, month - 1, month + 1):
        if BY_CM.get((c, mm)): return BY_CM[(c, mm)].most_common(1)[0][0]
    if BY_C.get(c): return BY_C[c].most_common(1)[0][0]
    return None


if __name__ == '__main__':
    O = pd.read_parquet('../data/op_results_2026.parquet')
    ev = O.drop_duplicates('tour')[['tour', 'cat', 'start']]
    ev['surf'] = [surface(a, b.month) for a, b in zip(ev.tour, ev.start)]
    print(ev.groupby('cat').surf.apply(lambda s: s.notna().mean()).round(3))
    print(ev.surf.value_counts())
    print(ev[ev.surf.isna()].tour.head(15).tolist())
