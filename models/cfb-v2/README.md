# CFB v2 in Bet Desk

J's v2 model (cfb_v2_model.py, on Drive in the CFB v2 folder) runs in the Bet Desk Games tab: CFB > "v2 (yours)".
The app has v2's projection and betting math built in (points from plays x PPA, sigma table, key numbers,
opening-line spread bands, 10.5-point totals rule, 1H/1Q shares). Its weekly team ratings need
CollegeFootballData, which is blocked from the cloud, so they come from J's computer:

1. Put `export_v2_ratings.py` (this folder) next to `cfb_v2_model.py`.
2. `pip install numpy pandas scipy scikit-learn requests` (once).
3. Set the key: `export CFBD_API_KEY=yourkey` (Windows: `set CFBD_API_KEY=yourkey`). Never paste it into the app or a file.
4. `python export_v2_ratings.py 2026 8` (season, the week you're betting). About 5 API calls after the first run.
5. Upload the `cfb_v2.json` it writes to the CFB v2 Drive folder and ask Claude to load it into Bet Desk
   (it is copied to app/data/cfb_v2.json and the app is republished).

Until a file is loaded, the v2 button shows these steps and v3 keeps working.
