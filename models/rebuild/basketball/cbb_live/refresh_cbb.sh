#!/bin/bash
# Daily in-season CBB v2 refresh (~25 min): hoopR box scores -> live copy of models/cbb (games, availability, ratings walk)
# -> rebuild team-games -> live_cbb.py -> app/data/cbb_v2.json. models/cbb itself is never touched: cbb/ here is a fresh copy.
set -e
cd "$(dirname "$0")"
rm -rf cbb && cp -r ../../../cbb cbb && rm -rf cbb/__pycache__
H=$(mktemp -d); B=https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb
for y in $(seq 2008 2027); do
  curl -sSfL -o $H/mbb_schedule_$y.parquet $B/schedules/parquet/mbb_schedule_$y.parquet
  curl -sSfL -o $H/team_box_$y.parquet $B/team_box/parquet/team_box_$y.parquet
  if [ $y -ge 2012 ]; then curl -sSfL -o $H/player_box_$y.parquet $B/player_box/parquet/player_box_$y.parquet; fi
done
cd cbb
CBB_LAST=2027 python3 prep_cbb.py $H data
CBB_LAST=2027 python3 prep_availability.py $H data
rm -rf $H
python3 game_model.py > run_log_live.txt 2>&1
cd ..
python3 build_tg.py
python3 ../live_cbb.py
