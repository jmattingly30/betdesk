"""Print the current CFB regular-season week (first week with unplayed games) from the cfbfastR 2026 schedule.

Retries with backoff (15, 30, 60, 120 s) because raw.githubusercontent.com sometimes answers 503 for a minute
(2026-11-19: one 503 left the morning chain's week empty and CFB v2 did not refresh). If every try fails it exits 1
with a clear message, so the step fails loudly instead of running with no week.
"""
import sys, time
import pandas as pd

URL = 'https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_2026.csv'
TRIES = 5

for i in range(1, TRIES + 1):
    try:
        s = pd.read_csv(URL)
        break
    except Exception as e:
        print(f'cfb_week: try {i}/{TRIES} failed: {e}', file=sys.stderr)
        if i == TRIES:
            sys.exit(f'cfb_week: could not read the CFB schedule after {TRIES} tries: {URL}')
        time.sleep(15 * 2 ** (i - 1))

s = s[(s.season_type == 'regular') & s.home_points.isna()]
print(int(s.week.min()) if len(s) else 0)
