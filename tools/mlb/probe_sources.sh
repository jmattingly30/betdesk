#!/usr/bin/env bash
# MLB source reachability probe (2026-11-24, thread "Move MLB model fully onto the desk").
# Fetches every URL J's MLB app (sources/baseball-win-tool) reads, plus free stand-ins for the FanGraphs
# uploads, and prints status / size / first line so we can see what a GitHub runner can reach.
# No secrets are printed. Odds API: /v4/sports only (costs 0 credits).
set -u
Y=${PROBE_SEASON:-2026}
D=${PROBE_DATE:-2026-09-20}
OUT=${PROBE_OUT:-probe_out}; mkdir -p "$OUT"
UA_CURL="curl/$(curl --version | head -1 | cut -d' ' -f2)"
probe() { # name url [ua]
  local name=$1 url=$2 ua=${3:-$UA_CURL}
  local code size ct
  read -r code size ct < <(curl -sS -L --max-time 60 -A "$ua" -o "$OUT/$name" -w '%{http_code} %{size_download} %{content_type}\n' "$url" 2>/dev/null || echo "ERR 0 -")
  local head1; head1=$(head -c 160 "$OUT/$name" 2>/dev/null | tr '\n\r\t' '   ')
  printf '%-28s %-4s %9s  %-32s %s\n' "$name" "$code" "$size" "${ct:0:32}" "$head1"
  echo "| $name | $code | $size | ${ct:0:40} |" >> "$OUT/summary.md"
}
echo "| source | http | bytes | type |" > "$OUT/summary.md"; echo "|---|---|---|---|" >> "$OUT/summary.md"
SV=https://baseballsavant.mlb.com
# --- what the app uses today ---
probe statsapi_schedule "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=$D&hydrate=team,probablePitcher,lineups,venue,weather"
probe statsapi_people   "https://statsapi.mlb.com/api/v1/people?personIds=592450,660271&hydrate=currentTeam"
probe savant_batters    "$SV/leaderboard/custom?year=$Y&type=batter&filter=&min=1&selections=pa,ab,woba,xwoba,k_percent,bb_percent,isolated_power&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&csv=true"
probe savant_pitchers   "$SV/leaderboard/custom?year=$Y&type=pitcher&filter=&min=1&selections=pa,woba,xwoba,p_formatted_ip&chart=false&x=pa&y=pa&r=no&chartType=beeswarm&csv=true"
probe savant_framing    "$SV/catcher_framing?year=$Y&team=&min=1&type=catcher&sort=4,1&csv=true"
probe savant_oaa        "$SV/leaderboard/outs_above_average?type=Fielder&startYear=$Y&endYear=$Y&split=no&team=&range=year&min=1&pos=&roles=&viz=hide&csv=true"
probe savant_pitchfile  "$SV/statcast_search/csv?all=true&hfSea=$Y%7C&hfGT=R%7C&player_type=pitcher&pitchers_lookup%5B%5D=669373&type=details&min_pitches=0&min_results=0&min_pas=0"
probe bbref_majors      "https://www.baseball-reference.com/leagues/majors/$Y.shtml"
probe bbref_majors_brUA "https://www.baseball-reference.com/leagues/majors/$Y.shtml" "Mozilla/5.0 (compatible; betdesk-probe)"
# --- FanGraphs (the app's hand uploads: reliever IP/TBF/wOBA, team wRC+, team FRV) ---
FG=https://www.fangraphs.com/api/leaders/major-league/data
probe fg_api_relievers  "$FG?age=&pos=all&stats=rel&lg=all&qual=0&season=$Y&season1=$Y&ind=0&team=0,ts&rost=0&month=0&pageitems=50&pagenum=1&type=8"
probe fg_api_team_bat   "$FG?age=&pos=all&stats=bat&lg=all&qual=0&season=$Y&season1=$Y&ind=0&team=0,ts&rost=0&month=0&pageitems=50&pagenum=1&type=8"
probe fg_api_team_fld   "$FG?age=&pos=all&stats=fld&lg=all&qual=0&season=$Y&season1=$Y&ind=0&team=0,ts&rost=0&month=0&pageitems=50&pagenum=1&type=1"
probe fg_leaders_page   "https://www.fangraphs.com/leaders/major-league?stats=rel&team=0%2Cts&season=$Y&season1=$Y&qual=0"
probe fg_guts           "https://www.fangraphs.com/guts.aspx?type=cn"
# --- free stand-ins ---
probe statsapi_team_rp  "https://statsapi.mlb.com/api/v1/teams/stats?stats=statSplits&group=pitching&sitCodes=rp&season=$Y&sportIds=1"
probe statsapi_team_bat "https://statsapi.mlb.com/api/v1/teams/stats?stats=season&group=hitting&season=$Y&sportIds=1"
probe savant_team_frv   "$SV/leaderboard/fielding-run-value?gameType=Regular&seasonStart=$Y&seasonEnd=$Y&type=team&position=&minInnings=0&minResults=1&csv=true"
probe savant_park       "$SV/leaderboard/statcast-park-factors?type=year&year=$Y&batSide=&stat=index_wOBA&condition=All&rolling=no&csv=true"
probe chadwick_people   "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-0.csv"
# --- odds feed (0-credit endpoint) ---
if [ -n "${ODDS_API_KEY:-}" ]; then
  code=$(curl -sS --max-time 30 -o "$OUT/odds_sports.json" -w '%{http_code}' "https://api.the-odds-api.com/v4/sports?apiKey=$ODDS_API_KEY" || echo ERR)
  mlb=$(grep -o '"key":"baseball_mlb[^"]*"' "$OUT/odds_sports.json" | tr '\n' ' ')
  printf '%-28s %-4s  %s\n' odds_api_sports "$code" "${mlb:-baseball_mlb not listed (offseason?)}"
  echo "| odds_api_sports | $code | ${mlb:-no baseball_mlb in list} | |" >> "$OUT/summary.md"
  rm -f "$OUT/odds_sports.json"
fi
cat "$OUT/summary.md" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
