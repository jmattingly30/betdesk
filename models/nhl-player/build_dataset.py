"""Build one row per player per regular-season/playoff game with pre-game features (only earlier games used).
usage: python3 build_dataset.py <dl> <out_dir>   -> <out_dir>/data/player_games.parquet"""
import sys, glob
import numpy as np, pandas as pd



def load_raw(dl):
    d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{dl}/pb_20*.parquet'))
                   if int(f[-12:-8]) >= 2024], ignore_index=True)
    return d[d.game_id.astype(str).str[4:6].isin(['02', '03'])].copy()


def build(d):
    """d: raw box rows; future games can be included with a `future` flag and empty stats."""
    d['playoff'] = (d.game_id.astype(str).str[4:6] == '03').astype(int)
    d['game_date'] = pd.to_datetime(d.game_date)
    mm = d.toi.fillna('0:00').str.split(':', expand=True)
    d['toi'] = mm[0].astype(float) + mm[1].astype(float) / 60
    d['is_g'] = d.saves.notna() | d.position.isna()
    d['position'] = np.where(d.is_g, 'G', d.position.fillna('C'))
    d['pos'] = d.position.map({'C': 'F', 'L': 'F', 'R': 'F', 'D': 'D', 'G': 'G'})
    fut = d['future'].fillna(False).astype(bool) if 'future' in d else pd.Series(False, index=d.index)
    d = d[(d.toi > 0) | fut].copy()
    for c in ['goals', 'assists', 'points', 'shots_on_goal', 'blocked_shots', 'hits', 'power_play_goals', 'pim', 'saves', 'shots_against', 'goals_against']:
        d[c] = d[c].astype(float)
    d = d.rename(columns={'shots_on_goal': 'sog', 'blocked_shots': 'blk', 'team_abbrev': 'team'})
    d['season'] = d.season.astype(int) // 10000 + 0  # start-year*1  (20252026 -> 2025)
    d = d.sort_values(['game_date', 'game_id', 'team', 'player_id']).reset_index(drop=True)

    # ---- team-game table (shots, goals for/against) and opponent ----
    tg = d[~d.is_g].groupby(['game_id', 'team', 'home_away', 'game_date', 'season', 'playoff']).agg(
        sf=('sog', 'sum'), gf=('goals', 'sum'), blk_t=('blk', 'sum'), hits_t=('hits', 'sum')).reset_index()
    opp = tg[['game_id', 'team', 'sf', 'gf', 'blk_t', 'hits_t']].rename(columns={'team': 'opp', 'sf': 'sa', 'gf': 'ga', 'blk_t': 'blk_a', 'hits_t': 'hits_a'})
    tg = tg.merge(opp, on='game_id'); tg = tg[tg.team != tg.opp].sort_values(['game_date', 'game_id']).reset_index(drop=True)
    # pre-game exponentially weighted team form (half-life 15 games, carries across seasons)
    def ewm_prev(s, hl):
        return s.groupby(level=0).transform(lambda x: x.shift().ewm(halflife=hl, min_periods=1).mean())
    tg = tg.set_index('team')
    for c in ['sf', 'sa', 'gf', 'ga', 'blk_t', 'hits_a']:
        tg[f'{c}_e'] = ewm_prev(tg[c], 15)
    tg = tg.reset_index()
    tg['rest'] = tg.groupby('team').game_date.diff().dt.days.fillna(4).clip(1, 7)
    lg = tg.groupby('season')[['sf', 'gf']].transform('mean')
    for c in ['sf_e', 'sa_e', 'gf_e', 'ga_e']:
        tg[c] = tg[c].fillna(lg[{'sf': 'sf', 'sa': 'sf', 'gf': 'gf', 'ga': 'gf'}[c[:2]]])
    tg['blk_t_e'] = tg.blk_t_e.fillna(tg.blk_t.mean()); tg['hits_a_e'] = tg.hits_a_e.fillna(tg.hits_a.mean())
    T = tg[['game_id', 'team', 'opp', 'rest', 'sf_e', 'sa_e', 'gf_e', 'ga_e', 'blk_t_e', 'hits_a_e', 'sa', 'sf']]
    O = tg[['game_id', 'team', 'sf_e', 'sa_e', 'gf_e', 'ga_e', 'rest', 'hits_a_e']].rename(columns=lambda c: 'o_' + c if c not in ('game_id',) else c).rename(columns={'o_team': 'opp'})
    d = d.merge(T, on=['game_id', 'team'], how='left').merge(O, on=['game_id', 'opp'], how='left')
    d['home'] = (d.home_away == 'home').astype(int)

    # ---- player pre-game history features ----
    d = d.sort_values(['player_id', 'game_date', 'game_id']).reset_index(drop=True)
    g = d.groupby('player_id')
    d['n_prev'] = g.cumcount()
    d['days_since'] = g.game_date.diff().dt.days.fillna(60).clip(0, 60)
    for hl in (3, 10):
        d[f'toi_e{hl}'] = g.toi.transform(lambda x: x.shift().ewm(halflife=hl, min_periods=1).mean())
    d['toi_last'] = g.toi.shift()
    STATS = ['sog', 'goals', 'assists', 'points', 'blk', 'hits', 'pim', 'power_play_goals']
    # position prior rates per 60 (from all seasons; small leakage-free alternative: previous seasons only -> use expanding by date)
    K = {'sog': 250, 'goals': 900, 'assists': 700, 'points': 600, 'blk': 250, 'hits': 200, 'pim': 400, 'power_play_goals': 1500}
    sk = ~d.is_g
    for s in STATS:
        # recency-weighted (half-life 25 games) sums of stat and TOI from earlier games
        num = g[s].transform(lambda x: x.shift().ewm(halflife=25, min_periods=1, adjust=True).mean())
        den = g['toi'].transform(lambda x: x.shift().ewm(halflife=25, min_periods=1, adjust=True).mean())
        w = (1 - 0.5 ** (d.n_prev / 25)) / (1 - 0.5 ** (1 / 25))   # effective number of games in the ewm
        d[f'{s}_raw60'] = (num / den * 60).where(d.n_prev > 0)
        # position average per 60 from the PREVIOUS season (first season: its own), so no future games leak in
        pr = d[sk].groupby(['season', 'pos'])[s].sum() / d[sk].groupby(['season', 'pos']).toi.sum() * 60
        prev = {(se, po): pr.get((se - 1, po), pr.get((se, po), pr.xs(po, level='pos').iloc[-1] if po in pr.index.get_level_values(1) else np.nan))
                for se, po in pr.index}
        for se in d.season.unique():
            for po in ('F', 'D'):
                prev.setdefault((se, po), pr.xs(po, level='pos').iloc[-1])
        d.loc[sk, f'{s}_pos60'] = [prev[(se, po)] for se, po in zip(d.season[sk], d.pos[sk])]
        eff_toi = (w * den).fillna(0)
        d[f'{s}_60'] = ((num.fillna(0) * w + K[s] / 60 * d[f'{s}_pos60']) / (eff_toi + K[s]) * 60).where(sk)
        d[f'{s}_e5'] = g[s].transform(lambda x: x.shift().ewm(halflife=5, min_periods=1).mean())
    # last-10 and season-to-date averages (baselines)
    for s in STATS + ['toi', 'saves', 'shots_against']:
        d[f'{s}_l10'] = g[s].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())
        d[f'{s}_szn'] = d.groupby(['player_id', 'season'])[s].transform(lambda x: x.shift().expanding().mean())
    # goalie save rate, shrunk to .900 with 600 shots of prior
    gk = d.is_g
    cs = d[gk].groupby('player_id').saves.transform(lambda x: x.shift().ewm(halflife=40, min_periods=1).mean())
    ca = d[gk].groupby('player_id').shots_against.transform(lambda x: x.shift().ewm(halflife=40, min_periods=1).mean())
    ng = d[gk].groupby('player_id').cumcount(); wg = (1 - 0.5 ** (ng / 40)) / (1 - 0.5 ** (1 / 40))
    d.loc[gk, 'svpct'] = ((cs.fillna(0) * wg + 0.900 * 600) / (ca.fillna(0) * wg + 600))
    d.loc[gk, 'g_starts_prev'] = ng
    d['started_g'] = gk & (d.toi >= 40)
    # share of the team's skater TOI (role)
    d['pos_code'] = d.position.map({'C': 0, 'L': 1, 'R': 1, 'D': 2, 'G': 3})
    keep = [c for c in d.columns if c not in ('home_away', 'sweater_number', 'faceoff_winning_pctg', 'plus_minus', 'shifts', 'giveaways', 'takeaways',
                                           'even_strength_shots_against', 'power_play_shots_against', 'shorthanded_shots_against', 'save_shots_against',
                                           'save_pctg', 'even_strength_goals_against', 'power_play_goals_against', 'shorthanded_goals_against', 'starter', 'decision')]
    return d[keep]


if __name__ == '__main__':
    dl, out = sys.argv[1], sys.argv[2]
    d = build(load_raw(dl))
    d.to_parquet(f'{out}/data/player_games.parquet', index=False)
    print(d.shape, d.season.value_counts().sort_index().to_dict())
