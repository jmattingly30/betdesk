#!/bin/bash
# Daily in-season CBB v2 refresh (~25 min): hoopR box scores -> live copy of models/cbb (games, availability, ratings walk)
# -> rebuild team-games -> live_cbb.py -> app/data/cbb_v2.json. models/cbb itself is never touched: cbb/ here is a fresh copy.
set -e
cd "$(dirname "$0")"
rm -rf cbb && cp -r ../../../cbb cbb && rm -rf cbb/__pycache__
H=$(mktemp -d); B=https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb
# get FILE URL: retries with backoff (1, 2, 4 ... 64 s; ~2 min) before failing, so one GitHub 503 doesn't sink the step
# (2026-11-19: a single 503 on one of ~50 hoopR files failed the morning run)
get(){ curl -sSfL --retry 7 --retry-all-errors -o "$1" "$2" || { echo "refresh_cbb: download failed after 7 retries: $2" >&2; exit 1; }; }
for y in $(seq 2008 2027); do
  get $H/mbb_schedule_$y.parquet $B/schedules/parquet/mbb_schedule_$y.parquet
  get $H/team_box_$y.parquet $B/team_box/parquet/team_box_$y.parquet
  if [ $y -ge 2012 ]; then get $H/player_box_$y.parquet $B/player_box/parquet/player_box_$y.parquet; fi
done
cd cbb
CBB_LAST=2027 python3 prep_cbb.py $H data
CBB_LAST=2027 python3 prep_availability.py $H data
rm -rf $H
python3 game_model.py > run_log_live.txt 2>&1 || { tail -n 30 run_log_live.txt; exit 1; }
cd ..
python3 build_tg.py
python3 ../live_cbb.py
