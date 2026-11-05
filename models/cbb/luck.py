"""Shooting-luck correction (2026-10-11, stat hunt): opponents' FT% and 3P% against a team are mostly luck, and so is a
team's own short-run 3P%/FT% swing; the ratings take those points at face value. Per team, over its last 10 games this
season (5+ needed): dluck = points opponents scored above league-average FT% and 3P% on their attempts, oluck = the same
for the team's own shooting. Walk-forward 2015-2026 (stat-hunt test): the model's margin miss moves +0.17 pts per point
of dluck and -0.10 per point of oluck; totals -0.12 per point of the four summed. Coefficients refit each season on
earlier seasons were stable (0.17-0.24, -0.10 to -0.16, -0.06 to -0.12). Gains: margin MAE 8.722 -> 8.701 (t 5.1),
ATS at 3+ pt gaps vs close 52.2% -> 53.2% (5,000 games); 2026 DK opener 3+ gaps CLV +1.55 -> +1.65, ATS 55.9% -> 58.0%.

team_luck() -> DataFrame team_id, dluck, oluck, n, m_adj, t_adj, dO, dD (per-100 rating shifts folded in by export_app)."""
import json, os, pandas as pd
here = os.path.dirname(os.path.abspath(__file__))

def team_luck(season=None, poss=68.0):
    k = json.load(open(f'{here}/model_config.json')).get('shooting_luck')
    t = pd.read_parquet(f'{here}/data/cbb_team_games.parquet')
    season = season or int(t.season.max()); t = t[t.season == season].sort_values('date')
    lg_ft, lg_3 = t.o_ftm.sum() / t.o_fta.sum(), t.o_tpm.sum() / t.o_tpa.sum()
    t['dluck'] = (t.o_ftm - t.o_fta * lg_ft) + 3 * (t.o_tpm - t.o_tpa * lg_3)
    t['oluck'] = (t.ftm - t.fta * lg_ft) + 3 * (t.tpm - t.tpa * lg_3)
    w = k['window']; g = t.groupby('team').tail(w).groupby('team').agg(dluck=('dluck', 'mean'), oluck=('oluck', 'mean'), n=('dluck', 'size'))
    g = g[g.n >= k['min_games']].copy()
    g['m_adj'] = k['def'] * g.dluck + k['off'] * g.oluck; g['t_adj'] = k['tot'] * (g.dluck + g.oluck)
    g['dO'] = 100 / poss * (g.m_adj + g.t_adj) / 2; g['dD'] = 100 / poss * (g.m_adj - g.t_adj) / 2
    return g.reset_index().rename(columns={'team': 'team_id'})
