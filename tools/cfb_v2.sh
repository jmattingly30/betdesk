#!/bin/bash
# CFB v2 weekly ratings (CFBD_API_KEY) -> app/data/cfb_v2.json -> public/cfb_v2.json
source "$(dirname "$0")/ci.sh"; setup_paths
WEEK="${CFB_WEEK:-}"
if [ -z "$WEEK" ]; then
  WEEK=$(python3 -c "import pandas as p;s=p.read_csv('https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_2026.csv');s=s[(s.season_type=='regular')&s.home_points.isna()];print(int(s.week.min()) if len(s) else 0)")
fi
sh_step "CFB v2 export week $WEEK" "cd models/cfb-v2 && python3 export_v2_ratings.py 2026 $WEEK && cp cfb_v2.json $SA/app/data/cfb_v2.json"
publish; commit "CFB v2 week $WEEK" || echo "commit" >> "$FAILS"; finish
