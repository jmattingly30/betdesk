#!/bin/bash
# Morning chain, per app/routines/morning_chain.md parts A-E. MLB (B7, Google Drive) is skipped: no Drive outside Claude.
source "$(dirname "$0")/ci.sh"; setup_paths
TODAY=$(TZ=$TZ_LOCAL date +%F); DOW=$(TZ=$TZ_LOCAL date +%u)   # 1=Mon .. 7=Sun
DL=/tmp/claude-0; mkdir -p $DL/nhldl $DL/nbadl $DL/nfldl
SW="${RUNNER_TEMP:-/tmp}/sweep.log"

# A. odds sweep (then tennis) in the background; B runs meanwhile
( cd "$SA/app" && python3 op_live.py NFL CFB NBA CBB NHL && python3 op_live.py TENNIS ) > "$SW" 2>&1 &
SWEEP=$!

# B. projections
step "B0 past props" python3 app/past_props.py
sh_step "B1 NHL props" "cd models/nhl-player && python3 fetch_data.py $DL/nhldl 2027 2027 && curl -sSfL -o $DL/nhldl/sched_2027.parquet https://github.com/sportsdataverse/sportsdataverse-data/releases/download/nhl_schedules/nhl_schedule_2027.parquet && python3 project.py $DL/nhldl $DL/nhl_props.json 2 && cd $SA && python3 app/add_recent.py NHL $DL/nhl_props.json && cp $DL/nhl_props.json app/data/nhl_props.json"
sh_step "B2 NBA props" "for y in \$(seq 2015 2027); do curl -sSfL -o $DL/nbadl/pb_\$y.parquet https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_nba_player_boxscores/player_box_\$y.parquet || exit 1; done; (cd models/nba-player && python3 build_dataset.py $DL/nbadl .) && python3 app/export_nba_props.py $TODAY $DL/nba_props.json && (python3 app/add_recent.py NBA $DL/nba_props.json || python3 app/add_recent.py NBA $DL/nba_props.json) && cp $DL/nba_props.json app/data/nba_props.json"
curl -sSfL -o $DL/nfldl/games.csv https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv || echo "nflverse games.csv" >> "$FAILS"
if [ "$DOW" != 1 ]; then
  WEEK=$(python3 -c "import pandas as p;g=p.read_csv('$DL/nfldl/games.csv');g=g[(g.season==2026)&(g.game_type=='REG')&g.result.isna()];print(int(g.week.min()))")
  sh_step "B3 NFL props (week $WEEK)" "(cd models/nfl-player && python3 fetch_data.py $DL/nfldl 2026 && python3 build_dataset.py $DL/nfldl .) && python3 app/export_nfl_props.py $DL/nfldl 2026 $WEEK $DL/nfl_props.json && (python3 app/add_recent.py NFL $DL/nfl_props.json || python3 app/add_recent.py NFL $DL/nfl_props.json) && cp $DL/nfl_props.json app/data/nfl_props.json"
fi
step "B3a NFL shadow grading" python3 models/nfl-player/shadow.py
step "B4 NFL team games" python3 app/export_games.py $DL/nfldl/games.csv
step "B5 CBB v2 (live_cbb)" python3 models/rebuild/basketball/live_cbb.py
if [ "$DOW" = 7 ]; then
  sh_step "B6 CFB v3 weekly rebuild" "cd models/cfb-v3 && curl -sSfL -o pbp_2026.parquet https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_cfb_pbp/play_by_play_2026.parquet && curl -sSfL -o sched_2026.csv https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/csv/cfb_schedules_2026.csv && D=../team-strength/data && python3 build.py \$D/cfb_games.parquet \$D/cfb_team_games_epa.csv \$D/cfb_returning.csv . pbp_2026.parquet sched_2026.csv --roster roster_prior.csv && python3 export_app.py"
fi
sh_step "B8 Model A NBA lineups + shadow" "cd models/rebuild/basketball && python3 live_lineup.py && cd ../shadow && DL=${RUNNER_TEMP:-/tmp} python3 shadow.py nba nfl nhl cbb"

# C. after the sweep
wait $SWEEP || echo "A op_live sweep" >> "$FAILS"
echo "::group::sweep log"; cat "$SW"; echo "::endgroup::"
grep -q "failed:" "$SW" && echo "A sweep printed failed: lines" >> "$FAILS"
step "C2 slates" bash -c "cd app && python3 export_slates.py"
step "C4 line tools" python3 app/line_tools.py
step "C4d CBB Model A+B" bash -c "python3 models/cbb_ab/live.py && python3 models/cbb_ab/grade.py"
step "C4e NFL Model A+B" bash -c "python3 models/nfl_ab/live.py && python3 models/nfl_ab/grade.py $DL/nfldl/games.csv"
step "C4f NBA Model A+B" bash -c "python3 models/nba_ab/live.py && python3 models/nba_ab/grade.py"
step "C4a CFB usage" bash -c "cd models/oddspapi && python3 cfb_usage.py"
step "C4b CFB Model A" bash models/rebuild/football/cfb_gap/live/run_live.sh
step "C4g CFB Model A+B" bash -c "python3 models/cfb_ab/live.py && python3 models/cfb_ab/grade.py"
step "C4c Model A export" python3 app/export_model_a.py
step "C4h Watch the sim" python3 models/sims/build.py
step "C6 parlays + promo" bash -c "python3 models/parlay/suggest.py && python3 models/promo/promo.py"
step "C7 snapshots" bash -c "node models/grading/snap_games.js && node models/grading/snap_flags.js"
step "C8 grade" python3 models/grading/grade.py

# D. publish = commit; E. report = the job fails if any step failed
publish; commit "morning chain" || echo "commit" >> "$FAILS"; finish
