"""Download (or refresh) the nflverse files the model needs into <dl_dir>.

python3 fetch_data.py <dl_dir> [SEASON]
Past seasons are downloaded once; the current SEASON (default 2026) and the schedule are always re-downloaded, so run
this before each week's projections (after the Friday injury reports for Sunday games)."""
import sys, os, urllib.request

dl = sys.argv[1]
cur = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
os.makedirs(dl, exist_ok=True)
B = 'https://github.com/nflverse/nflverse-data/releases/download'

def get(url, path, force=False):
    if os.path.exists(path) and not force:
        return
    try:
        urllib.request.urlretrieve(url, path + '.tmp'); os.replace(path + '.tmp', path); print('ok  ', path)
    except Exception as e:
        print('FAIL', url, e)

for y in range(2016, cur + 1):
    f = y == cur
    get(f'{B}/pbp/play_by_play_{y}.parquet', f'{dl}/pbp_{y}.parquet', f)
    get(f'{B}/snap_counts/snap_counts_{y}.parquet', f'{dl}/snaps_{y}.parquet', f)
get(f'{B}/injuries/injuries_{cur}.parquet', f'{dl}/inj_{cur}.parquet', True)
get(f'{B}/players/players.parquet', f'{dl}/players.parquet', True)
for k in ('rushing', 'receiving'):     # NFL-3 team-changer target model (2026-10-10)
    get(f'{B}/nextgen_stats/ngs_{k}.csv.gz', f'{dl}/ngs_{k}.csv.gz', True)
get('https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv', f'{dl}/games.csv', True)
