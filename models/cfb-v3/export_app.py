"""Write the CFB v3 ratings + pricing constants for the Bet Desk Games tab. python3 export_app.py [out.json]"""
import sys, os, json, datetime, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1] if len(sys.argv) > 1 else f'{here}/../../app/data/cfb_model.json'
cfg = json.load(open(f'{here}/model_config.json')); meta = json.load(open(f'{here}/ratings_meta.json'))
r = pd.read_csv(f'{here}/ratings_current.csv').sort_values('team')
# CFB-2 (2026-10-10): the roster/transfer terms of the margin stack are a per-team amount (home minus away), so they are
# folded into each team's margin rating m (divided by the stack's m coefficient). The app's formula is unchanged.
if cfg.get('roster_file'):
    st = cfg['stack_margin']['late']; RP = pd.read_csv(f"{here}/{cfg['roster_file']}"); RP = RP[RP.season == meta['season']].set_index('team_id')
    RP['qb_q'] = RP.qb_anya + 0.3 * RP.qb_lvl * (1 - RP.qb_none); dflt = cfg['roster_defaults']
    # 2026 transfer volumes run past anything the stack was fit or tested on (98th pct 0.94 in 2025, Auburn 1.97 in 2026),
    # so cap them at the tested range instead of extrapolating
    RP['xfer_skill'] = RP.xfer_skill.clip(upper=1.0); RP['avail_skill'] = RP.avail_skill.clip(upper=2.4)
    def adj(tid):
        x = RP.loc[tid] if tid in RP.index else pd.Series(dflt)
        return sum(st[f'r_{k}'] * float(x.get(k, dflt[k])) for k in dflt)
    a = r.id.map(adj); r['m'] = r.m + (a - a.mean()) / st['m_m']
# market rating (J 2026-10-26, season-adjustments card 1): the stack's mk term is mk[home] - mk[away] + mk_hfa*home, so the
# team part folds into m (divided by the stack's m coefficient) and the home part into the stack's hf weight. The app's
# margin formula is unchanged.
stack = json.loads(json.dumps(cfg['stack_margin']))
if 'mk' in stack['late'] and 'mk' in r:
    for k in stack:
        stack[k]['hf'] += stack[k]['mk'] * cfg['mk_par']['hfa']
    st = stack['late']; r['m'] = r.m + st['mk'] * r.mk.fillna(0) / st['m_m']
doc = dict(made=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC'), through=meta['through'], hfa=meta['hfa'], mu_p=meta['mu_p'],
           stack=stack, stack_total=cfg['stack_total'], sd_margin=cfg['sd_margin'],
           open=dict(b=cfg['b_open'], s=cfg['sd_open'], bt=cfg['bt_open'], st=cfg['sdt_open']), close=dict(b=cfg['b_close'], s=cfg['sd_close'], bt=cfg['bt_close'], st=cfg['sdt_close']),
           teams=[[t.team, int(t.fbs), round(t.m, 3), round(t.o, 4), round(t.d, 4), round(t.op, 3), round(t.dp, 3)] for t in r.itertuples()])
json.dump(doc, open(out, 'w'), separators=(',', ':'), allow_nan=False); print(len(doc['teams']), 'teams ->', out)
