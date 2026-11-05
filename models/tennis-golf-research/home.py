"""Host country for 2026 events (from event name) and player nationality (deuce/TML player files, by name)."""
import pandas as pd
from bridge import norm
C2IOC = {'ARG': 'ARG', 'Argentina': 'ARG', 'Australia': 'AUS', 'Austria': 'AUT', 'Bahrain': 'BRN', 'Belgium': 'BEL', 'Belgium Singles': 'BEL',
         'Bolivia': 'BOL', 'Brazil': 'BRA', 'Bulgaria': 'BUL', 'Canada': 'CAN', 'Chile': 'CHI', 'China': 'CHN', 'Colombia': 'COL',
         'Cote d Ivoire': 'CIV', 'Croatia': 'CRO', 'Czech Republic': 'CZE', 'Dominican Republic': 'DOM', 'Ecuador': 'ECU', 'Egypt': 'EGY',
         'Finland': 'FIN', 'France': 'FRA', 'Germany': 'GER', 'Great Britain': 'GBR', 'Greece': 'GRE', 'Hong Kong': 'HKG', 'India': 'IND',
         'Ireland': 'IRL', 'Italy': 'ITA', 'Jamaica': 'JAM', 'Japan': 'JPN', 'Kazakhstan': 'KAZ', 'Korea Republic': 'KOR', 'Mexico': 'MEX',
         'Moldova': 'MDA', 'Monaco': 'MON', 'Morocco': 'MAR', 'Netherlands': 'NED', 'New Caledonia': 'FRA', 'New Zealand': 'NZL',
         'Paraguay': 'PAR', 'Philippines': 'PHI', 'Poland': 'POL', 'Portugal': 'POR', 'Qatar': 'QAT', 'Republic of Congo': 'CGO',
         'Romania': 'ROU', 'Rwanda': 'RWA', 'San Marino': 'SMR', 'Singapore': 'SGP', 'Slovakia': 'SVK', 'Slovenia': 'SLO',
         'South Africa': 'RSA', 'Spain': 'ESP', 'Sweden': 'SWE', 'Switzerland': 'SUI', 'Thailand': 'THA', 'Tunisia': 'TUN',
         'Turkey': 'TUR', 'Turkiye': 'TUR', 'UAE': 'UAE', 'USA': 'USA', 'Vietnam': 'VIE'}
SLAMS = {'Australian Open': 'AUS', 'French Open': 'FRA', 'Wimbledon': 'GBR', 'US Open': 'USA'}


def host(t):
    for k, v in SLAMS.items():
        if t.startswith(k): return v
    if ',' not in t: return None
    c = t.split(',')[1].replace('Men Singles', '').replace('Women Singles', '').strip()
    return C2IOC.get(c)


def nationality(tour):
    p = pd.read_parquet(f'../data/{tour}_players.parquet')
    p = p[p.player_id != 'player_id']
    d = dict(zip((p.first_name.astype(str) + ' ' + p.last_name.astype(str)).map(norm), p.country_code))
    if tour == 'atp':
        t = pd.read_csv('../../dl/Tennismylife_TML-Database_master_2025.csv')
        d.update(dict(zip(t.winner_name.map(norm), t.winner_ioc))); d.update(dict(zip(t.loser_name.map(norm), t.loser_ioc)))
    return d
