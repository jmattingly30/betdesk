#!/bin/bash
# CBB v2 refresh on its own (morning chain B5, then the CBB steps that read it), for re-running after a failed morning.
source "$(dirname "$0")/ci.sh"; setup_paths
step "B5 CBB v2 (in-season ratings refresh + live_cbb)" bash models/rebuild/basketball/cbb_live/refresh_cbb.sh
step "C4d CBB Model A+B" bash -c "python3 models/cbb_ab/live.py && python3 models/cbb_ab/grade.py"
step "C4c Model A export" python3 app/export_model_a.py
step "C4h Watch the sim" python3 models/sims/build.py CBB
publish; commit "CBB v2 refresh" || echo "commit" >> "$FAILS"; finish
