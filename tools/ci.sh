#!/bin/bash
# Shared helpers for the GitHub Actions pipeline. Source it: `source tools/ci.sh`.
# The scripts were written for the Claude project folder, so the repo is linked at that path and every
# routine step from app/routines/*.md runs unchanged. A failed step is recorded and the job carries on,
# so whatever did refresh still gets published; `finish` then fails the job (GitHub emails you).
set -u
export SA="${SA:-/mnt/project-files/sports-analytics}" TZ_LOCAL=America/Denver
FAILS="${RUNNER_TEMP:-/tmp}/betdesk_fails.txt"; : > "$FAILS"

setup_paths(){
  sudo mkdir -p /mnt/project-files /tmp/claude-0 && sudo chmod 777 /tmp/claude-0
  [ -e "$SA" ] || sudo ln -s "$GITHUB_WORKSPACE" "$SA"
  rm -f "$SA"/app/data/odds/*.lock
}

# step "label" command...   (runs in $SA unless the command cds)
step(){
  local label="$1"; shift
  echo "::group::$label"
  local out="${RUNNER_TEMP:-/tmp}/step.log"
  if (cd "$SA" && "$@") 2>&1 | tee "$out"; [ "${PIPESTATUS[0]}" = 0 ]; then echo "ok: $label"
  else echo "::error::FAILED: $label"; { echo "== $label"; tail -n 12 "$out" | cut -c1-300; } >> "$FAILS"; fi
  echo "::endgroup::"
}
sh_step(){ step "$1" bash -c "$2"; }

# Copy the data files the page reads from app/data into public/ (replaces the Artifact publish).
publish(){
  local D="$SA/app/data" P="$SA/public" n=0 f b
  # crossed games/teams/players fail the job (GitHub emails you); the files still publish, as with any failed step
  step "data check (games, teams, players line up)" python3 tools/check_data.py app/data
  for f in "$D"/*.json "$D"/odds/props_NFL.json "$D"/odds/props_NBA.json "$D"/odds/props_NHL.json "$D"/odds/tennis.json; do
    b=$(basename "$f")
    case "$b" in line_history.json|mlb_slate*.json|odds_test_*.json|cfb_live.json) continue ;; esac   # inputs the page never reads
    [ -f "$f" ] && ! cmp -s "$f" "$P/$b" && cp "$f" "$P/$b" && n=$((n+1))
  done
  cmp -s "$SA/app/sim.js" "$P/sim.js" || { cp "$SA/app/sim.js" "$P/sim.js"; n=$((n+1)); }   # Watch the sim viewer (loaded on tap)
  echo "published $n changed data files to public/"
}

commit(){
  cd "$GITHUB_WORKSPACE"
  git config user.name "betdesk-bot"; git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  git add -A
  if git diff --cached --quiet; then echo "nothing to commit"; return 0; fi
  git commit -q -m "data: $1 $(TZ=$TZ_LOCAL date '+%Y-%m-%d %H:%M %Z')"
  for i in 1 2 3 4; do git pull -q --rebase -X theirs origin main && git push -q origin HEAD:main && return 0; sleep $((2**i)); done
  echo "::error::push failed"; return 1
}


finish(){
  if [ -s "$FAILS" ]; then echo "::error::failed steps:"; cat "$FAILS"; exit 1; fi
  echo "all steps ok"
}
