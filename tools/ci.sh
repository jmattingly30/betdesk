#!/bin/bash
# Shared helpers for the GitHub Actions pipeline. Source it: `source tools/ci.sh`.
# The scripts were written for the Claude project folder, so the repo is linked at that path and every
# routine step from app/routines/*.md runs unchanged. A failed step is recorded and the job carries on,
# so whatever did refresh still gets published; `finish` then fails the job (GitHub emails you).
set -u
export SA=/mnt/project-files/sports-analytics TZ_LOCAL=America/Denver
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
  if (cd "$SA" && "$@"); then echo "ok: $label"; else echo "::error::FAILED: $label"; echo "$label" >> "$FAILS"; fi
  echo "::endgroup::"
}
sh_step(){ step "$1" bash -c "$2"; }

# Copy the data files the page reads from app/data into public/ (replaces the Artifact publish).
publish(){
  local D="$SA/app/data" P="$SA/public" n=0
  for f in past_props.json nhl_props.json nba_props.json nfl_props.json games_models.json cbb_v2.json cfb_model.json cfb_v2.json \
           game_slates.json line_tools.json parlay_suggestions.json promo_picks.json scorecard.json player_results.json \
           cfb_usage.json model_a.json cbb_ab.json tennis_model.json props_vs_close.json landing.json; do
    [ -f "$D/$f" ] && ! cmp -s "$D/$f" "$P/$f" && cp "$D/$f" "$P/$f" && n=$((n+1))
  done
  for f in props_NFL.json props_NBA.json props_NHL.json tennis.json; do
    [ -f "$D/odds/$f" ] && ! cmp -s "$D/odds/$f" "$P/$f" && cp "$D/odds/$f" "$P/$f" && n=$((n+1))
  done
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
