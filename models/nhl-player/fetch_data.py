"""Download NHL player box scores (sportsdataverse-data releases, NHL API source) into <dl>.
Only 2023-24 onward carry game ids and dates, so earlier seasons are not used.
usage: python3 fetch_data.py <dl> [first_season_end_year=2024] [last=2027]"""
import sys, os, urllib.request
dl = sys.argv[1]; first = int(sys.argv[2]) if len(sys.argv) > 2 else 2024; last = int(sys.argv[3]) if len(sys.argv) > 3 else 2027
os.makedirs(dl, exist_ok=True)
for y in range(first, last + 1):
    url = f'https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nhl_player_boxscores/player_box_{y}.parquet'
    urllib.request.urlretrieve(url, f'{dl}/pb_{y}.parquet'); print('got', y)
