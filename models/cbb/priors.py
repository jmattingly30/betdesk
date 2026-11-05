"""Preseason ratings from last season, two seasons ago, and roster turnover (returning minutes, transfers in).

For each season y the coefficients are fit only on seasons before y (expanding window), so the prior is a true forecast.
Targets are end-of-season opponent-adjusted O, D (points per 100 possessions vs an average team) and tempo P."""
import os, numpy as np, pandas as pd

FEATS = ['lag1', 'lag2', 'lag1_x_ret', 'ret', 'xfer_min', 'xfer_q', 'new_d1']
# CBB-2 (2026-10-10): player-built inputs from data/cbb_player_seasons.csv (prep_player_prior.py); CBB_PLAYER_PRIOR=0 turns them off
PLAYER = os.environ.get('CBB_PLAYER_PRIOR', '1') != '0' and os.path.exists('data/cbb_player_seasons.csv')
if PLAYER:
    FEATS = FEATS + ['r_val', 'r_val_lvl', 'xfer_val']
    PS = pd.read_csv('data/cbb_player_seasons.csv'); REPL = float(open('data/cbb_player_repl.txt').read())

def team_features(final, links):
    """final: {season: DataFrame(index=team, O, D, P, net)}; links: roster links table."""
    rows = []
    seasons = sorted(final)
    extra = [seasons[-1] + 1] if (links.season == seasons[-1] + 1).any() else []   # next season: features only (no target yet)
    for y in seasons[1:] + extra:
        f1 = final[y - 1]; f2 = final.get(y - 2)
        f0 = final[y] if y in final else pd.DataFrame({c: np.nan for c in ['O', 'D', 'P']},
                                                      index=pd.Index(sorted(set(links.team[links.season == y]) | set(f1.index)), name=f1.index.name))
        L = links[links.season == y]
        net_prev = f1.net
        L = L.assign(prev_net=L.prev_team.map(net_prev))
        if PLAYER:
            pv = PS[PS.season == y - 1].sort_values('games').drop_duplicates('athlete_id', keep='last').set_index('athlete_id')
            L = L.assign(mpg=L.athlete_id.map(pv.mpg).fillna(0.0), val=L.athlete_id.map(pv.value).fillna(REPL))
            L = L.assign(contrib=L.mpg * (L.val - REPL) * 40, lvl=L.prev_net.fillna(-10.0) / 10)
        agg = L.groupby('team').apply(lambda x: pd.Series({
            'ret': x.share[x.returning].sum(),
            'xfer_min': x.share[~x.returning].fillna(0).sum(),
            'xfer_q': (x.share * x.prev_net)[~x.returning].fillna(0).sum(),
            **({'r_val': (x.contrib * min(1.0, 200 / max(x.mpg.sum(), 1))).sum(),
                'r_val_lvl': (x.contrib * x.lvl * min(1.0, 200 / max(x.mpg.sum(), 1))).sum(),
                'xfer_val': (x.contrib * ~x.returning * min(1.0, 200 / max(x.mpg.sum(), 1))).sum()} if PLAYER else {})}))
        for comp in ['O', 'D', 'P']:
            df = pd.DataFrame(index=f0.index)
            df['y'] = f0[comp]
            df['lag1'] = f1[comp].reindex(df.index)
            df['lag2'] = (f2[comp].reindex(df.index) if f2 is not None else np.nan)
            df = df.join(agg)
            df['new_d1'] = df.lag1.isna().astype(float)
            df['lag2'] = df.lag2.fillna(df.lag1)
            df[['lag1', 'lag2']] = df[['lag1', 'lag2']].fillna(0)
            df['ret'] = df.ret.fillna(df.ret.mean()) - 0.4
            df['lag1_x_ret'] = df.lag1 * df.ret
            df[['xfer_min', 'xfer_q']] = df[['xfer_min', 'xfer_q']].fillna(0)
            if PLAYER: df[['r_val', 'r_val_lvl', 'xfer_val']] = df[['r_val', 'r_val_lvl', 'xfer_val']].fillna(0)
            if comp == 'D': df['xfer_q'] = df.xfer_q  # a better source team helps both ends; sign is learned
            df['season'] = y; df['comp'] = comp; df['team'] = df.index
            rows.append(df.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)

def fit_priors(final, links, first_season=2011, verbose=False):
    """Returns {season: DataFrame(index=team, O, D, P)} for seasons > first_season, plus coefficient table."""
    X = team_features(final, links)
    out, coefs = {}, []
    seasons = sorted(X.season.unique())
    for y in seasons:
        if y < first_season: continue
        pr = {}
        for comp in ['O', 'D', 'P']:
            tr = X[(X.season < y) & (X.comp == comp)].dropna(subset=['y'])
            te = X[(X.season == y) & (X.comp == comp)]
            A = np.c_[tr[FEATS].values, np.ones(len(tr))]
            beta = np.linalg.lstsq(A, tr.y.values, rcond=None)[0]
            pr[comp] = pd.Series(np.c_[te[FEATS].values, np.ones(len(te))] @ beta, index=te.team.values)
            coefs.append(dict(season=y, comp=comp, **dict(zip(FEATS + ['const'], beta.round(3)))))
        out[y + 1 - 1] = pd.DataFrame(pr)
    return out, pd.DataFrame(coefs), X
