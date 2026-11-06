#!/bin/bash
# Bring the repo up to date with the Claude project folder (run from a Claude session that has the folder).
# Usage: bash tools/sync_from_project.sh [PROJECT_DIR]   (default /mnt/project-files/sports-analytics)
# 1. code + model files: every file the repo already tracks under app/ models/ op/ archive/ is copied if it changed;
#    new code files (.py .js .sh .md .json under 5 MB) in folders the repo already tracks, and new models/*_ab folders, are added
# 2. app/data/*.json: copied when the project copy is newer than the repo's last commit of that file
# 3. public/index.html rebuilt from app/bet-desk.html (tools/standalone_patch.py), data copied into public/
# Then review `git status` and commit.
set -eu
P="${1:-/mnt/project-files/sports-analytics}"; R="$(cd "$(dirname "$0")/.." && pwd)"; cd "$R"
n=0
while IFS= read -r f; do
  case "$f" in app/data/*) continue ;; esac
  if [ -f "$P/$f" ] && ! cmp -s "$P/$f" "$f"; then cp "$P/$f" "$f"; n=$((n+1)); echo "updated $f"; fi
done < <(git ls-files app models op archive)
dirs=$( (git ls-files models app | xargs -n1 dirname | sort -u; cd "$P" && ls -d models/*_ab 2>/dev/null) | sort -u)
for d in $dirs; do
  [ -d "$P/$d" ] || continue
  find "$P/$d" -maxdepth 1 -type f \( -name '*.py' -o -name '*.js' -o -name '*.sh' -o -name '*.md' -o -name '*.json' \) -size -5M | while read -r s; do
    f="${s#$P/}"; case "$f" in app/data/*|*.lock) continue ;; esac
    [ -e "archive/applied-patches/$(basename "$f")" ] && [ "${f%/*}" = app ] && continue   # applied one-shot patches, archived
    [ -e "$f" ] || { mkdir -p "$(dirname "$f")"; cp "$s" "$f"; echo "added $f"; }
  done
done
for s in "$P"/app/data/*.json "$P"/app/data/odds/*.json "$P"/app/data/odds/op/*.json; do
  f="${s#$P/}"; [ -f "$s" ] || continue
  last=$(git log -1 --format=%ct -- "$f" 2>/dev/null || true); last=${last:-0}
  if [ "$(stat -c %Y "$s")" -gt "$last" ] && ! cmp -s "$s" "$f" 2>/dev/null; then mkdir -p "$(dirname "$f")"; cp "$s" "$f"; echo "data $f"; fi
done
python3 tools/standalone_patch.py app/bet-desk.html public/index.html
SA="$R" bash -c 'source tools/ci.sh; publish'
echo "updated $n tracked files; now: git status, then commit and push"
