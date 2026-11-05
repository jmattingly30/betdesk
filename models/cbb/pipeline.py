"""Shared loading + two-pass ratings for game_model.py and tune.py (2026-10-09 audit version)."""
import numpy as np, pandas as pd
import ratings, priors

def load():
    g = pd.read_parquet('data/cbb_games.parquet'); tg = pd.read_parquet('data/cbb_team_games.parquet'); g = g[g.d1].copy()
    v = pd.read_parquet('data/cbb_venue.parquet')
    g = g.merge(v, on='game_id', how='left')
    links = pd.read_parquet('data/cbb_roster_links.parquet')
    return g, tg, links

def add_loc(g, nh):
    """Home-team location value: 1 at a true home game; at a 'neutral' site, +nh if only the home-listed team is in its
    home state, -nh if only the away team is, else 0."""
    hin = (g.venue_state == g.home_state_h) & g.venue_state.notna()
    ain = (g.venue_state == g.home_state_a) & g.venue_state.notna()
    g = g.copy()
    g['loc_h'] = np.where(~g.neutral, 1.0, nh * (hin.astype(float) - ain.astype(float)))
    return g

def run(g, tg, links, params, nh=0.0, verbose=False):
    g = add_loc(g, nh)
    _, final0 = ratings.walk_forward(g, tg, **params)
    pr, coefs, X = priors.fit_priors(final0, links)
    h, final = ratings.walk_forward(g, tg, priors=pr, **params)
    return h, final, pr, coefs
