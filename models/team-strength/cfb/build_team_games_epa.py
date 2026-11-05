"""Aggregate ESPN/cfbfastR CFB play-by-play to team-game offensive efficiency (EPA/play, success rate,
non-garbage EPA). Usage: build_team_games_epa.py <pbp parquet> ... -> appends to <out csv>."""
import sys, os, numpy as np, pandas as pd, pyarrow.parquet as pq
out = sys.argv[-1]
cols = ['season','game_id','pos_team_id','def_pos_team_id','EPA','EPA_success','wp_before','scrimmage_play','pass','rush','week']
for f in sys.argv[1:-1]:
    have = set(pq.read_schema(f).names)
    p = pq.read_table(f, columns=[c for c in cols if c in have]).to_pandas()
    p = p[(p.scrimmage_play == True) & p.EPA.notna()]
    p['ng'] = p.wp_before.between(0.1, 0.9)
    k = ['season','game_id','pos_team_id','def_pos_team_id']
    gb = p.groupby(k)
    t = pd.DataFrame({'plays': gb.size(), 'epa': gb.EPA.mean(), 'sr': gb.EPA_success.mean(),
                      'pass_epa': p[p['pass'] == 1].groupby(k).EPA.mean(), 'rush_epa': p[p['rush'] == 1].groupby(k).EPA.mean(),
                      'ng_epa': p[p.ng].groupby(k).EPA.mean(), 'ng_sr': p[p.ng].groupby(k).EPA_success.mean()}).reset_index()
    t.to_csv(out, mode='a', header=not os.path.exists(out), index=False)
    print(f, len(t))
