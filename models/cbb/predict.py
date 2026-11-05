"""Price a college basketball game: projected score, possessions, win probability, cover and over probabilities.

python3 predict.py "Duke" "North Carolina"                        # first team is home
python3 predict.py "Duke" "Houston" --neutral --line -2.5 --total 131.5 --at open
python3 predict.py "Duke" "Houston" --preseason                   # 2026-27 preseason ratings (no roster info yet)

--line is the HOME team's spread in betting notation (home -2.5 = home favoured by 2.5); --at open|close picks which
market-blend weight to use (open lines are softer, so more of the model's disagreement counts).

2026-10-09 audit:
  --out-home "Name;Name2" / --out-away: players not playing, priced from each player's saved value (C1). Only list
      players whose absence the market may not have priced yet; a player out for weeks is already in the team rating.
  --near-home home|away: a 'neutral' site in that team's home state gets part of a home edge (C5; 0 if tuning dropped it)
  --month 11..3: sigma by month and pace (C3); default = this month (November before the season)
  The opener weight is the cross-fit one, shrunk toward the close weight (C2), not the old in-sample 0.35.
2026-10-10 (CBB-1 + CBB-2): at --at open the weight is per conference tier of the game's lower team (tier_split.py).
2026-10-11 (stat hunt): in-season ratings get the shooting-luck correction (luck.py: opponents' and own FT%/3P% luck over
the last 10 games, folded into O and D). Not applied to --preseason ratings (no games yet); --no-luck turns it off."""
import argparse, datetime, json, os, re, sys, unicodedata, numpy as np, pandas as pd
from scipy.stats import norm
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
cfg = json.load(open(f'{here}/model_config.json'))

def preseason_ratings():
    """Next season's prior: latest expanding-window coefficients on last two seasons, average roster turnover."""
    r = pd.read_csv(f'{here}/data/cbb_ratings_by_season.csv'); co = pd.read_csv(f'{here}/prior_coefficients.csv')
    y = r.season.max(); c = co[co.season == co.season.max()].set_index('comp')
    ret = pd.read_csv(f'{here}/data/cbb_returning.csv'); last = ret[ret.season == ret.season.max()]
    fp = f'{here}/data/cbb_prior_feature_means.csv'; fm = pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()
    a, b = r[r.season == y].set_index('team'), r[r.season == y - 1].set_index('team')
    out = pd.DataFrame(index=a.index); out['name'] = a.name
    for comp in ['O', 'D', 'P']:
        k = c.loc[comp]; lag1 = a[comp]; lag2 = b[comp].reindex(a.index).fillna(lag1); rt = last.ret_min.mean() - 0.4
        out[comp] = k.lag1*lag1 + k.lag2*lag2 + k.lag1_x_ret*lag1*rt + k.ret*rt + k.xfer_min*last.xfer_min.mean() + k.const
        # roster terms not known before rosters post (xfer_q, CBB-2 player values): last season's league average
        for f_ in ['xfer_q', 'r_val', 'r_val_lvl', 'xfer_val']:
            if f_ in k.index and f_ in fm.columns: out[comp] += k[f_] * fm[f_].iloc[-1]
    out['net'] = out.O + out.D; out['mu'] = a.mu.iloc[0]; out['hca'] = a.hca.iloc[0]; out['T0'] = a.T0.iloc[0]
    out['tempo'] = out.T0 + out.P
    return out.reset_index().rename(columns={'team': 'team_id'})

def apply_luck(r):
    """Shift O and D by each team's shooting-luck correction (luck.py) when the config has it and the team has 5+ games."""
    if 'shooting_luck' not in cfg: return r
    from luck import team_luck
    g = team_luck().set_index('team_id'); r = r.copy()
    r['O'] = r.O + r.team_id.map(g.dO).fillna(0); r['D'] = r.D + r.team_id.map(g.dD).fillna(0); r['net'] = r.O + r.D
    return r

def team_tiers():
    p_ = f'{here}/data/cbb_team_tier.csv'
    return pd.read_csv(p_).set_index('team_id').tier.to_dict() if os.path.exists(p_) else {}

def find(r, q):
    m = r[r.name.str.lower() == q.lower()]
    if m.empty: m = r[r.name.str.lower().str.contains(q.lower(), regex=False)]
    if len(m) != 1: raise SystemExit(f'"{q}" matched {len(m)} teams: {", ".join(m.name.head(8))}')
    return m.iloc[0]

def _norm(x):
    x = unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.sub(r'\b(jr|sr|ii|iii|iv)\b', '', re.sub(r'[^a-z ]', '', x)).split())

def out_value(team_id, names):
    """Sum of saved player values (points of margin) for the listed players of this team."""
    if not names: return 0.0, []
    pv = pd.read_csv(f'{here}/data/cbb_player_values.csv'); pv = pv[pv.team_id == team_id]; pv['key'] = pv.name.map(_norm)
    tot, found = 0.0, []
    for n_ in [x for x in names.split(';') if x.strip()]:
        m_ = pv[pv.key == _norm(n_)]
        if m_.empty: m_ = pv[pv.key.str.contains(_norm(n_), regex=False)]
        if len(m_) != 1: print(f'  {n_}: {len(m_)} matches on this team, ignored'); continue
        tot += max(0.0, float(m_.value.iloc[0])); found.append(f'{m_.name.iloc[0]} ({m_.value.iloc[0]:.1f})')
    return tot, found

def price(r, home, away, neutral=False, line=None, total=None, at='close', out_home='', out_away='', near_home=None, month=None):
    h, a = find(r, home), find(r, away)
    loc = 1.0 if not neutral else (cfg.get('near_home_neutral', 0.0) * {'home': 1, 'away': -1}.get(near_home, 0))
    mu, hca, T0 = h.mu, h.hca * loc, h.T0
    poss = h.tempo + a.tempo - T0
    ph = (mu + h.O - a.D + hca) * poss / 100; pa = (mu + a.O - h.D - hca) * poss / 100
    av = cfg.get('availability', {}); vh, fh = out_value(h.team_id, out_home); va, fa = out_value(a.team_id, out_away)
    dm = av.get('k_margin', 0.0) * (va - vh); dt = av.get('k_total', 0.0) * (va + vh)
    ph, pa = ph + dm / 2 + dt / 2, pa - dm / 2 + dt / 2
    m, t = ph - pa, ph + pa
    mon = month or datetime.date.today().month
    mi = min(5, max(1, mon - 10 if mon >= 10 else mon + 2))
    sig = cfg['sigma_by_month'][str(mi)] * (poss / 68) ** cfg.get('sigma_pace_exp', 0.0) if 'sigma_by_month' in cfg else cfg['sigma_margin']
    scale = sig / cfg['sigma_margin']
    out = {'home': h['name'], 'away': a['name'], 'possessions': round(poss, 1), 'home_pts': round(ph, 1), 'away_pts': round(pa, 1),
           'margin': round(m, 1), 'total': round(t, 1), 'sigma': round(sig, 2), 'p_home_win': round(norm.cdf(m / sig), 3)}
    if fh or fa: out['players out (value)'] = '; '.join(fh + fa)
    bo, bc = cfg['blend_open'], cfg['blend_close']
    b, s, bt, st = (bo['b_spread_open'], bo['sd_ats_open'], bo['b_total_open'], bo['sd_total_open']) if at == 'open' else (bc['b_spread'], bc['sd_ats'], bc['b_total'], bc['sd_total'])
    tw = cfg.get('blend_open_tier')
    if at == 'open' and tw:   # tier_split.py: the opener is softer vs the model in high-major games, firmer in low-major
        tt = team_tiers(); rank = {'high': 0, 'mid': 1, 'low': 2}
        g_t = max([h.team_id, a.team_id], key=lambda i: rank[tt.get(i, 'low')]); out['tier'] = tt.get(g_t, 'low'); b = tw[out['tier']]
    if line is not None:
        exp = -line; gap = m - exp
        out.update({'market_margin': exp, 'model_minus_market': round(gap, 1), 'blended_margin': round(exp + b*gap, 1),
                    'p_home_cover': round(norm.cdf(b*gap / (s * scale)), 3)})
    if total is not None:
        gap = t - total; out.update({'model_minus_total': round(gap, 1), 'p_over': round(norm.cdf(bt*gap / (st * scale)), 3)})
    return out

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('home'); p.add_argument('away'); p.add_argument('--neutral', action='store_true')
    p.add_argument('--line', type=float); p.add_argument('--total', type=float); p.add_argument('--at', choices=['open', 'close'], default='open')
    p.add_argument('--preseason', action='store_true'); p.add_argument('--out-home', default=''); p.add_argument('--out-away', default='')
    p.add_argument('--near-home', choices=['home', 'away']); p.add_argument('--month', type=int); p.add_argument('--no-luck', action='store_true'); x = p.parse_args()
    r = preseason_ratings() if x.preseason else pd.read_csv(f'{here}/ratings_cbb.csv')
    if not x.preseason and not x.no_luck: r = apply_luck(r)
    for k, v in price(r, x.home, x.away, x.neutral, x.line, x.total, x.at, x.out_home, x.out_away, x.near_home, x.month).items(): print(f'{k:>20}: {v}')
