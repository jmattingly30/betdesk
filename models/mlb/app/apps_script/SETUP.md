# MLB Apps Script: put the newest version in place (2026-10-19)

Best done on a computer's browser. About 10 minutes. Your web app link stays the same.

## The four files (newest copies, all in this folder)
| File | Last changed | What's new |
|---|---|---|
| `Server.gs` | 2026-10-19 (late) | Your efficiency model is primary in the slate (v2 removed; v3 kept for comparison). Slate export for Bet Desk (`exportSlate`, `installSlateTimer`) + last-3-days `finals` for model grading; each final now carries the game's last pre-first-pitch projection (`pregame`) |
| `Bundle.gs` | 2026-10-19 (late) | Model code: your efficiency model (season R/G at full strength, last season's efficiency for games 1-29, then games/162) is primary and sets picks; v3, v3 + defense, v3 full logged for comparison; model v2 retired; your sheet models unchanged |
| `Index.html` | 2026-10-19 (late) | The app page: efficiency model on top, comparison models folded away. Copy it from `Index.html.txt` (same code; the .html file shows blank in a file viewer) |
| `appsscript.json` | 2026-10-17 | Adds full Drive permission (`auth/drive`, changed 2026-10-19 because `drive.file` could not find the folder) so the app can write `mlb_slate.json` to your Drive |

Checked 2026-10-19: `Bundle.gs`/`Index.html` match a fresh build from `src/`, both `.gs` files parse,
Apps Script tests 45/45, phone UI 57/57 and model tests 210/210 pass.

## Steps
1. Open https://script.google.com and open your project **"MLB Win Probability"**.
2. If you can't see `appsscript.json` in the file list: **Project Settings** (gear) -> tick
   **Show "appsscript.json" manifest file in editor**, then back to **Editor** (`<>`).
3. For each file, click it in the editor, select all (Ctrl+A), and paste the new contents:
   - `appsscript.json` <- `appsscript.json`
   - `Server.gs` <- `Server.gs`
   - `Bundle.gs` <- `Bundle.gs`
   - `Index.html` <- `Index.html`
4. Save (Ctrl+S).
5. In the function menu at the top pick **`installSlateTimer`** and click **Run**. Google asks for permission again
   (new: "see, edit, create files in Google Drive that you use with this app"). Allow it.
   This sets the 30-minute timer and writes the first `mlb_slate.json` into a Drive folder named **"Bet Desk MLB"**
   (created if missing).
6. Check it worked: open the Execution log. You should see a line like
   `mlb_slate.json: 15 games (12 run, 0 left for the next pass)`. On an off day it will say 0 games, which is fine.
   Optional: run **`exportSlate`** any time to refresh by hand.
7. **Deploy -> Manage deployments -> pencil -> Version: New version -> Deploy.** The link doesn't change.
   No need to re-run `setup`.

## After it's in place
- Bet Desk's MLB slate reads `mlb_slate.json` (stale after 2 h), and model grading reads the `finals` in it. Each final's `pregame` is the last
  projection the timer made before first pitch (kept 7 days in Script Properties as `slate_pre_<gamePk>`; null if the game was never run).
- Big slates may take two passes: games not reached within ~4.5 min show as "pending" and are done on the next timer run.
- If a data source (Savant framing/OAA) blocks Google's servers, the log names the site; the efficiency model still runs.

Full background (setup from scratch, tests, privacy, limits): `../README.md`, section
"Phone version (Google Apps Script)". Bet Desk side of the feed: `sports-analytics/app/README.md`, "MLB full model feed".
