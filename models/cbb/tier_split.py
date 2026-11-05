"""CBB opener edge by conference tier, and per-tier opener weights (2026-10-10, J's mid-major focus).

cd models/cbb && python3 tier_split.py [base_predictions.parquet]
Tiers rank conferences by their members' mean end-of-season net rating: top 5 = high-major, next 10 = mid-major,
rest = low-major. A game takes the LOWER tier of its two teams. Prints opener accuracy, how much of the model's gap to
the open shows up in results (b), the |gap| >= 3 rule and CLV, by tier, conference vs non-conference and month;
with a base file it prints the same for that model. Then fits one opener weight per tier (cross-fit by date halves,
shrunk halfway to the close weight like game_model.py) and saves them to model_config.json 'blend_open_tier' and team
tiers to data/cbb_team_tier.csv. Log: tier_split_log.txt."""
import sys, json, numpy as np, pandas as pd, statsmodels.api as sm
log = open('tier_split_log.txt', 'w')
def say(*a): s = ' '.join(str(x) for x in a); print(s); log.write(s + '\n')
cfg = json.load(open('model_config.json')); Y = cfg['season']
G = pd.read_parquet('data/cbb_games.parquet', columns=['season', 'home', 'away', 'home_conference_id', 'away_conference_id'])
conf = pd.concat([G[['season', 'home', 'home_conference_id']].set_axis(['season', 'team', 'conf'], axis=1),
                  G[['season', 'away', 'away_conference_id']].set_axis(['season', 'team', 'conf'], axis=1)])
conf = conf.dropna().groupby(['season', 'team']).conf.agg(lambda s: s.mode().iloc[0])
def load(p):
    d = pd.read_parquet(p); return d[(d.season == Y) & d.line_open.notna() & d.line_close_dk.notna()].copy()
N = load('data/cbb_predictions.parquet')
R = pd.read_csv('ratings_cbb.csv').set_index('team_id') if 'team_id' in pd.read_csv('ratings_cbb.csv', nrows=1) else None
r = pd.concat([N[['date', 'home', 'O_h', 'D_h']].set_axis(['date', 'team', 'O', 'D'], axis=1),
               N[['date', 'away', 'O_a', 'D_a']].set_axis(['date', 'team', 'O', 'D'], axis=1)]).sort_values('date').groupby('team').last()
r['net'] = r.O + r.D; r['conf'] = r.index.map(lambda t: conf.get((Y, t)))
cs = r.groupby('conf').net.mean().sort_values(ascending=False)
tier = {c: ('high' if i < 5 else 'mid' if i < 15 else 'low') for i, c in enumerate(cs.index)}
say(f'{Y} conferences by mean net rating (ESPN conference id: net):', ', '.join(f'{int(c)}:{v:.1f}' for c, v in cs.items()))
team_tier = pd.Series({t: tier.get(conf.get((Y, t)), 'low') for t in set(N.home) | set(N.away) | set(r.index)})
RANK = {'high': 0, 'mid': 1, 'low': 2}; NAME = {0: 'high', 1: 'mid', 2: 'low'}
def tag(d):
    th, ta = d.home.map(team_tier).fillna('low'), d.away.map(team_tier).fillna('low')
    d['tier'] = np.maximum(th.map(RANK), ta.map(RANK)).map(NAME); d['conf_game'] = th.index.map(lambda i: True) & (d.home.map(lambda t: conf.get((Y, t))) == d.away.map(lambda t: conf.get((Y, t))))
    d['g'] = d.margin_hat - d.line_open; d['y'] = d.margin - d.line_open; d['move'] = d.line_close_dk - d.line_open
    return d
N = tag(N)
rm = lambda x: float(np.sqrt((x ** 2).mean()))
def stats(d, thr=3):
    s = d[d.g.abs() >= thr]; pk = np.sign(s.g); c = np.sign(s.y) * pk; w, l = int((c > 0).sum()), int((c < 0).sum())
    b = sm.OLS(d.y, sm.add_constant(d.g)).fit()
    return dict(games=len(d), open_rmse=rm(d.y), close_rmse=rm(d.margin - d.line_close_dk), model_rmse=rm(d.margin - d.margin_hat),
                b_vs_open=b.params.iloc[1], t=b.tvalues.iloc[1], bets_gap3=w + l, ats_gap3=w / (w + l) if w + l else np.nan, clv_gap3=float((pk * s.move).mean()))
def table(d, lab):
    rows = [dict(model=lab, split=f'{t}-major game' if t == 'high' else f'{t}-major involved', **stats(d[d.tier == t])) for t in ['high', 'mid', 'low']]
    rows += [dict(model=lab, split='conference game' if c else 'non-conference', **stats(d[d.conf_game == c])) for c in [True, False]]
    mo = pd.to_datetime(d.date).dt.month.isin([11, 12])
    rows += [dict(model=lab, split='Nov-Dec', **stats(d[mo])), dict(model=lab, split='Jan-Mar', **stats(d[~mo]))]
    return rows
rows = table(N, 'current')
if len(sys.argv) > 1: rows += table(tag(load(sys.argv[1])), 'base')
say('\nOpener (DK) by split. b_vs_open = share of the model-minus-open gap that showed up in results\n' + pd.DataFrame(rows).round(3).to_string())
# per-tier opener weights, cross-fit by date halves
N = N.sort_values('date'); N['half'] = np.where(np.arange(len(N)) < len(N) // 2, 'A', 'B')
bc = cfg['blend_close']['b_spread']; cf = {'one': [], **{t: [] for t in RANK}}
for a_, b_ in [('A', 'B'), ('B', 'A')]:
    tr, te = N[N.half == a_], N[N.half == b_]
    b1 = np.polyfit(tr.g, tr.y, 1)[0]; bt = {t: np.polyfit(x.g, x.y, 1)[0] for t, x in tr.groupby('tier')}
    cf['one'].append(b1); [cf[t].append(v) for t, v in bt.items()]
    p1, pt = b1 * te.g, te.tier.map(bt) * te.g
    def rule(p):
        s = te[p.abs() >= 1.0]; c = np.sign(s.y) * np.sign(p[s.index]); return f'{len(s)} bets {100 * (c > 0).sum() / max(1, (c != 0).sum()):.1f}% CLV {(np.sign(p[s.index]) * s.move).mean():+.2f}'
    say(f'cross-fit {a_}->{b_}: open RMSE {rm(te.y):.3f} | one weight {b1:.2f}: {rm(te.y - p1):.3f}, blended edge >= 1 pt: {rule(p1)} | '
        f'per tier ' + ' '.join(f'{t} {v:.2f}' for t, v in bt.items()) + f': {rm(te.y - pt):.3f}, {rule(pt)}')
w = {t: round(float(0.5 * (np.mean(cf[t]) + bc)), 4) for t in RANK}
say(f'saved per-tier opener weights (cross-fit, shrunk halfway to the close weight {bc:.3f}):', w, f'| single weight stays {cfg["blend_open"]["b_spread_open"]:.3f}')
cfg['blend_open_tier'] = w; json.dump(cfg, open('model_config.json', 'w'), indent=1)
T = pd.read_csv('ratings_cbb.csv')[['team_id', 'name']]; T['tier'] = T.team_id.map(team_tier).fillna('low'); T['conf_id'] = T.team_id.map(lambda t: conf.get((Y, t)))
T.to_csv('data/cbb_team_tier.csv', index=False); say('team tiers:', T.tier.value_counts().to_dict())
