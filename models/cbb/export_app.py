"""Export CBB team ratings + pricing constants for the Bet Desk app (CBB game card on the Price tab).

python3 export_app.py [--preseason] [out.json]
--preseason: next season's prior ratings (no shooting-luck correction: no games yet) (use until about 8 games in; after that rebuild with the new season's
games, see README, and export without the flag). Default out: ../../app/data/cbb_ratings.json"""
import sys, os, json, datetime, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import preseason_ratings, cfg, here, team_tiers, apply_luck
pre = '--preseason' in sys.argv
args = [a for a in sys.argv[1:] if a != '--preseason']
out = args[0] if args else f'{here}/../../app/data/cbb_ratings.json'
r = preseason_ratings() if pre else apply_luck(pd.read_csv(f'{here}/ratings_cbb.csv'))   # luck correction in-season only
r = r.dropna(subset=['O', 'D', 'tempo']).sort_values('name'); tt = team_tiers()
doc = dict(made=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
           kind='2026-27 preseason prior (last two seasons + average roster turnover)' if pre else f"end of season {cfg['season']} ratings",
           mu=round(float(r.mu.iloc[0]), 3), hca=round(float(r.hca.iloc[0]), 3), T0=round(float(r.T0.iloc[0]), 3),
           sigma_by_month=cfg['sigma_by_month'], sigma_margin=cfg['sigma_margin'], pace_exp=cfg.get('sigma_pace_exp', 0.0),
           open=dict(b=cfg['blend_open']['b_spread_open'], s=cfg['blend_open']['sd_ats_open'], bt=cfg['blend_open']['b_total_open'], st=cfg['blend_open']['sd_total_open']),
           close=dict(b=cfg['blend_close']['b_spread'], s=cfg['blend_close']['sd_ats'], bt=cfg['blend_close']['b_total'], st=cfg['blend_close']['sd_total']),
           open_tier=cfg.get('blend_open_tier'),   # opener weight by the game's lower tier; teams[i][4] = tier (h/m/l)
           teams=[[n, round(float(o), 2), round(float(d), 2), round(float(t), 2), tt.get(i, 'low')[0]] for n, o, d, t, i in zip(r.name, r.O, r.D, r.tempo, r.team_id)])
json.dump(doc, open(out, 'w'), separators=(',', ':'), allow_nan=False)
print(len(doc['teams']), 'teams ->', out, doc['kind'])
