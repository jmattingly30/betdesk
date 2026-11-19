#!/bin/bash
# CFB v2 weekly ratings (CFBD_API_KEY) -> app/data/cfb_v2.json -> public/cfb_v2.json
source "$(dirname "$0")/ci.sh"; setup_paths
WEEK="${CFB_WEEK:-}"
if [ -z "$WEEK" ]; then
  WEEK=$(cd "$SA" && python3 tools/cfb_week.py) || { echo "== CFB v2: could not read the CFB schedule after 5 tries (see log)" >> "$FAILS"; finish; }
fi
sh_step "CFB v2 export week $WEEK" "cd models/cfb-v2 && python3 export_v2_ratings.py 2026 $WEEK && cp cfb_v2.json $SA/app/data/cfb_v2.json"
publish; commit "CFB v2 week $WEEK" || echo "commit" >> "$FAILS"; finish
