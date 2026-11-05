"""Which devig method gives the most accurate fair chance on OUR graded data? (Model B, J 2026-10-30)

python3 devig_test.py          # -> results/devig_test.csv, results/devig_cal.json, printed summary

Data (all graded, two-way):
  props   OddsPapi 2026 player props, Pinnacle / DraftKings / FanDuel, open and close (models/oddspapi/results/*_joined)
  nba_ml  SBR NBA closing moneylines 2011-2021 (one consensus-ish book)
  mlb_ml  MLB DraftKings moneylines 2021-2025, open and close (models/team-strength/data/mlb_odds_games.csv)
Score: log loss (lower = better) for every method, paired standard error vs power, plus a logistic fit
y ~ a + b*logit(p_power): a = side bias left after devig (a<0 on overs = book shades overs), b > 1 = devig
under-corrects the favourite-longshot bias, b < 1 = over-corrects. 'calibrated' = (a, b) fitted on the first half of
dates, scored on the second half (honest out-of-sample)."""
import os, sys, json, math
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devig as D

HERE = os.path.dirname(os.path.abspath(__file__)); R = f'{HERE}/../oddspapi/results'; OUT = f'{HERE}/results'
os.makedirs(OUT, exist_ok=True)
M = ['mult', 'add', 'power', 'shin', 'odds_ratio', 'mpto']

def fit_logit(x, y, iters=30):
    X = np.c_[np.ones_like(x), x]; w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (y - p); H = (X * (p * (1 - p))[:, None]).T @ X
        w += np.linalg.solve(H + 1e-9 * np.eye(2), g)
    return w

def ll(p, y): p = np.clip(p, 1e-6, 1 - 1e-6); return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def score(name, o1, o2, y, date):
    """o1/o2 decimal odds of side 1/2, y = 1 if side 1 won."""
    ok = (o1 > 1) & (o2 > 1) & np.isfinite(o1) & np.isfinite(o2) & ((1 / o1 + 1 / o2) > 0.98) & ((1 / o1 + 1 / o2) < 1.25)
    o1, o2, y, date = o1[ok], o2[ok], y[ok].astype(float), date[ok]
    if len(y) < 200: return None
    P = {m: np.array([D.METHODS[m]([a, b])[0] for a, b in zip(o1, o2)]) for m in M}
    L = {m: ll(P[m], y) for m in M}
    row = dict(set=name, n=len(y), hold=float(np.mean(1 / o1 + 1 / o2 - 1)), side1_won=float(y.mean()),
               side1_power=float(P['power'].mean()))
    for m in M:
        row[m] = float(L[m].mean())
        d = L[m] - L['power']; row[m + '_vs_power_se'] = float(d.std() / math.sqrt(len(d)))
    a, b = fit_logit(np.array([D.logit(p) for p in P['power']]), y); row['a'], row['b'] = float(a), float(b)
    # out-of-sample calibrated devig: fit on earlier half of dates, test on later half
    order = np.argsort(date, kind='stable'); h = len(y) // 2; tr, te = order[:h], order[h:]
    lp = np.array([D.logit(p) for p in P['power']])
    a2, b2 = fit_logit(lp[tr], y[tr]); pc = 1 / (1 + np.exp(-(a2 + b2 * lp[te])))
    row['cal_test'] = float(ll(pc, y[te]).mean()); row['power_test'] = float(L['power'][te].mean())
    row['cal_vs_power_se'] = float((ll(pc, y[te]) - L['power'][te]).std() / math.sqrt(len(te)))
    row['cal_a'], row['cal_b'] = float(a2), float(b2)
    return row

def main():
    rows = []
    # ---- props ----
    for sp, f in (('NBA', 'nba_joined.parquet'), ('NHL', 'nhl_joined.parquet'), ('NFL', 'live_nfl_joined.parquet')):
        d = pd.read_parquet(f'{R}/{f}')
        if 'did_not_play' in d: d = d[~d.did_not_play.fillna(False).astype(bool)]
        d = d[d.actual.notna() & (d.actual != d.line)]
        dt = pd.to_datetime(d['date'] if 'date' in d else d.get('game_date', d['fid'])).values if ('date' in d or 'game_date' in d) else d.fid.values
        d = d.assign(_dt=dt, y=(d.actual > d.line).astype(int))
        for bk, g in d.groupby('book'):
            for t in ('open', 'close'):
                r = score(f'{sp} props {bk} {t}', g[f'o_{t}'].values.astype(float), g[f'u_{t}'].values.astype(float), g.y.values, g._dt.values)
                if r: r.update(sport=sp, kind='props', book=bk, time=t); rows.append(r)
            # alt/longshot lines only (over price >= 2.5 or <= 1.4): where devig methods disagree most
            far = g[(g.o_close >= 2.5) | (g.o_close <= 1.4)]
            r = score(f'{sp} props {bk} close, lopsided', far.o_close.values.astype(float), far.u_close.values.astype(float), far.y.values, far._dt.values)
            if r: r.update(sport=sp, kind='props_lopsided', book=bk, time='close'); rows.append(r)

    # ---- NBA moneylines (SBR archive) ----
    s = pd.DataFrame(json.load(open(f'{HERE}/../nba-team/data/sbr_nba_archive_10Y.json')))
    s = s[pd.to_numeric(s.home_close_ml, errors='coerce').notna() & pd.to_numeric(s.away_close_ml, errors='coerce').notna()]
    s = s[(s.home_close_ml.abs() >= 100) & (s.away_close_ml.abs() >= 100)]
    s['hf'] = pd.to_numeric(s.home_final, errors='coerce'); s['af'] = pd.to_numeric(s.away_final, errors='coerce'); s = s[s.hf != s.af]
    o1 = s.home_close_ml.map(D.dec_from_am).values; o2 = s.away_close_ml.map(D.dec_from_am).values
    r = score('NBA ML close (SBR 2011-21)', o1, o2, (s.hf > s.af).astype(int).values, s.date.values)
    if r: r.update(sport='NBA', kind='ml', book='sbr', time='close'); rows.append(r)

    # ---- MLB moneylines (DraftKings) ----
    m = pd.read_csv(f'{HERE}/../team-strength/data/mlb_odds_games.csv'); m = m[m.hs != m.as_]
    for t in ('open', 'close'):
        mm = m[m[f'dk_home_{t}'].notna() & m[f'dk_away_{t}'].notna()]
        r = score(f'MLB ML DK {t} (2021-25)', mm[f'dk_home_{t}'].map(D.dec_from_am).values, mm[f'dk_away_{t}'].map(D.dec_from_am).values,
                  (mm.hs > mm.as_).astype(int).values, mm.date.values)
        if r: r.update(sport='MLB', kind='ml', book='draftkings', time=t); rows.append(r)

    T = pd.DataFrame(rows); T.to_csv(f'{OUT}/devig_test.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    show = ['set', 'n', 'hold', 'side1_won', 'side1_power'] + M + ['a', 'b', 'power_test', 'cal_test', 'cal_vs_power_se']
    print(T[show].round(4).to_string(index=False))
    best = T[M].idxmin(axis=1); print('\nBest method by set:'); print(pd.concat([T.set, best], axis=1).to_string(index=False))
    # wins counted by weighted n
    print('\nTotal log loss x n (lower better):', {m: round(float((T[m] * T.n).sum()), 1) for m in M})
    cal = {f"{r.book}|{r.kind}|{r.sport}|{r.time}": [r.a, r.b] for r in T.itertuples()}
    json.dump(cal, open(f'{OUT}/devig_cal.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
