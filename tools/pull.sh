#!/bin/bash
# Noon / night / NFL pulls, per app/routines/odds_pulls.md. Usage: tools/pull.sh noon|night|nfl
source "$(dirname "$0")/ci.sh"; T="$1"; setup_paths
case "$T" in
  noon)  step "op_live NFL NBA NHL (groups 1-3)" bash -c "cd app && OP_GROUPS=3 python3 op_live.py NFL NBA NHL" ;;
  night) step "op_live all sports (groups 1-3)"  bash -c "cd app && OP_GROUPS=3 python3 op_live.py NFL CFB NBA CBB NHL"
         step "op_live TENNIS"                    bash -c "cd app && python3 op_live.py TENNIS" ;;
  nfl)   step "op_live NFL (all books)"           bash -c "cd app && python3 op_live.py NFL" ;;
  *) echo "unknown pull $T"; exit 2 ;;
esac
if [ "$T" = noon ]; then step "slates (Odds API college lines)" bash -c "cd app && ODDS_FORCE=CFB,CBB python3 export_slates.py"
else step "slates (no Odds API credits)" bash -c "cd app && env -u ODDS_API_KEY python3 export_slates.py"; fi
step "line tools" bash -c "cd app && python3 line_tools.py"
if [ "$T" != nfl ]; then step "CBB Model B (cbb_ab live)" python3 models/cbb_ab/live.py; fi
step "NFL Model B (nfl_ab live)" python3 models/nfl_ab/live.py
if [ "$T" = night ]; then step "CFB usage" bash -c "cd models/oddspapi && python3 cfb_usage.py"; fi
publish; commit "$T pull" || echo "commit" >> "$FAILS"; finish
