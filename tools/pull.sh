#!/bin/bash
# Noon / night / NFL pulls, per app/routines/odds_pulls.md. Usage: tools/pull.sh noon|night|nfl
source "$(dirname "$0")/ci.sh"; T="$1"; setup_paths
case "$T" in
  # 2026-11-20 (J): props from OddsPapi's live endpoint, Pinnacle + DraftKings, 2 billed requests per pull (budget-paced);
  # game lines from The Odds API, which only the morning chain (all sports) and NFL pulls (NFL) spend credits on
  noon)  step "OddsPapi live props NFL NBA NHL" bash -c "cd app && python3 op_live.py --live NFL NBA NHL" ;;
  night) step "OddsPapi live props NFL NBA NHL CFB" bash -c "cd app && python3 op_live.py --live NFL NBA NHL CFB" ;;
  nfl)   step "OddsPapi live props NFL" bash -c "cd app && python3 op_live.py --live NFL" ;;
  *) echo "unknown pull $T"; exit 2 ;;
esac
step "injury scan (NBA Model A sees news since the morning)" python3 models/injuries/scan.py
if [ "$T" != nfl ]; then step "MLB slate before the game slates (lineups posted since the morning; finals for grading)" node models/mlb/live.js; fi
if [ "$T" = nfl ]; then step "slates (Odds API NFL lines: 3 credits)" bash -c "cd app && ODDS_FORCE=NFL python3 export_slates.py"
else step "slates (no Odds API credits; morning lines reused)" bash -c "cd app && env -u ODDS_API_KEY python3 export_slates.py"; fi
step "line tools" bash -c "cd app && python3 line_tools.py"
if [ "$T" != nfl ]; then step "CBB Model B (cbb_ab live)" python3 models/cbb_ab/live.py; fi
step "NFL Model B (nfl_ab live)" python3 models/nfl_ab/live.py
if [ "$T" != nfl ]; then step "NBA Model B (nba_ab live)" python3 models/nba_ab/live.py; fi
if [ "$T" != nfl ]; then step "NHL sim (re-logged on the new lines)" python3 models/nhl_sim/live.py ${RUNNER_TEMP:-/tmp}; fi
if [ "$T" = night ]; then step "CFB usage" bash -c "cd models/oddspapi && python3 cfb_usage.py"; fi
if [ "$T" != nfl ]; then step "CFB Model B (cfb_ab live)" python3 models/cfb_ab/live.py; fi
# Tennis Model A+B and grading held (J 2026-11-11: tennis off the desk; models/tennis_ab kept for later)
step "Guest prop picks (headless desk)" node models/grading/snap_flags.js --picks-only
if [ "$T" = nfl ]; then step "Watch the sim (NFL)" python3 models/sims/build.py NFL; else step "Watch the sim" python3 models/sims/build.py NBA CBB NFL CFB; fi
publish; commit "$T pull" || echo "commit" >> "$FAILS"; finish
