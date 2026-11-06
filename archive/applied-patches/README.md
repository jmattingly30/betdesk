# Applied one-shot patches (archived 2026-11-06 audit)

Every script here was a one-time edit to `app/bet-desk.html` and is already applied. Do not run them again: their anchors
have moved, and `best_bets.js`, `line_shop.js` and `prop_odds.js` are OLDER copies of code that now lives (edited since)
inside `app/bet-desk.html`. The desk's source is `app/bet-desk.html`; `app/redesign.js` + `app/redesign_inline.py` is the
only live inline flow. `tools/sync_from_project.sh` skips these names so they are not copied back into `app/`.
