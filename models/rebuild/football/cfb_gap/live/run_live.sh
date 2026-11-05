#!/bin/bash
# CFB Model A refresh: schedule + 2026 pbp (sportsdataverse) -> tempo -> desk probe -> cfb_live.py -> app/data/cfb_live.json
set -e
cd "$(dirname "$0")"
T=$(mktemp -d)
curl -sSfL -o cfbros/s_2026.csv https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_2026.csv
curl -sSfL -o $T/espn_2026.parquet https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_pbp/play_by_play_2026.parquet
python3 cfb_pace_prep.py $T 2026
python3 ../../../../cfb_ab/usage_prep.py $T 2026   # CFB Model A missing-star table (J 2026-11-05)
rm -rf $T
node ../../../../promo/probe_games.js probe.json
python3 cfb_live.py
