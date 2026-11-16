"""NHL shot-state sim: team-game tables, pre-game ratings and the frame the simulator reads.
Ratings as tuned in the October research (half-life 80 team-games, 0.3 carried into a new season, shrink 0.5);
each channel = (k + sum w*observed) / (k + sum w*expected), expected from league rates by score and skater counts.
Rest: a team on the second night of a back-to-back has its scoring rate multiplied by REST_MULT."""
import json, numpy as np, pandas as pd

HL, CARRY, KS = 80, 0.3, 0.5
REST_MULT = 0.92          # second night of a back-to-back; fitted in backtest.py (0.917 on 1,675 team-games)
PARAMS = {"scale": 1.04, "rate_sd": 0.10, "hfa": 1.03, "pen_mult": 1.0, "ot_mult": 1.0}
PPM = ["5v4", "5v3", "4v3"]; PKM = ["4v5", "3v5", "3v4"]
KEY = ["game_id", "season", "side", "team", "opp"]
CH = {"off_fen": ("fen", "e_fen", 300 * KS), "def_fen": ("o_fen", "o_e_fen", 300 * KS),
      "off_q": ("xg", "e_xg_given_fen", 25 * KS), "def_q": ("o_xg", "o_e_xg_given_fen", 25 * KS),
      "pp": ("ppxg", "e_ppxg", 8 * KS), "pk": ("o_ppxg", "o_e_ppxg", 8 * KS),
      "pen_taken": ("pk_t", "e_pk", 1500 * KS), "pen_drawn": ("o_pk_t", "o_e_pk", 1500 * KS),
      "fin": ("g_ne", "xg_all", 200 * KS)}
GK_K = 120 * KS


def _reg(d):
    return (~d.pull) & (~d.opull) & (d.pb != "ot")


def sums(s, t):
    """League sums per season that the base rates are made of (kept as sums so two seasons can be pooled)."""
    s = s[s.unbl == 1]; ne = s[~s.opull]
    T5 = t[_reg(t) & (t.man == "5v5")]; S5 = s[_reg(s) & (s.man == "5v5")]
    TM = t[_reg(t)]; SM = s[_reg(s)]
    tt = TM.assign(pk=TM.man.isin(PKM))
    f = lambda x: {f"{int(a)}|{b}": float(v) for (a, b), v in x.items()}
    return {"goals": {str(int(k)): float(v) for k, v in ne.groupby("season").goal.sum().items()},
            "xg": {str(int(k)): float(v) for k, v in ne.groupby("season").xg.sum().items()},
            "t5": f(T5.groupby(["season", "sd"]).dt.sum()), "n5": f(S5.groupby(["season", "sd"]).size()),
            "q5": f(S5.groupby(["season", "sd"]).xg.sum()),
            "tm": f(TM.groupby(["season", "man"]).dt.sum()), "qm": f(SM.groupby(["season", "man"]).xg.sum()),
            "tpk": f(tt[tt.pk].groupby(["season", "sd"]).dt.sum()), "tall": f(tt.groupby(["season", "sd"]).dt.sum())}


def pool(S, season, prev_w=0.0):
    """Sums for one season, with the season before mixed in at weight prev_w (used early in a live season)."""
    out = {}
    for name, tab in S.items():
        o = {}
        for k, v in tab.items():
            p = k.split("|"); yr = int(p[0])
            if yr == season: o["|".join(p[1:])] = o.get("|".join(p[1:]), 0) + v
            elif yr == season - 1 and prev_w > 0: o["|".join(p[1:])] = o.get("|".join(p[1:]), 0) + prev_w * v
        out[name] = o
    return out


def base_of(P):
    """Base rates for one season from its (pooled) sums. xG is rescaled so league xG equals league goals."""
    sc = P["goals"][""] / P["xg"][""]
    return {"scale": sc,
            "b_fen": {k: P["n5"][k] / P["t5"][k] for k in P["n5"]},
            "b_q": {k: sc * P["q5"][k] / P["n5"][k] for k in P["n5"]},
            "b_xgm": {k: sc * P["qm"].get(k, 0.0) / v for k, v in P["tm"].items() if v > 0},
            "b_pk": {k: P["tpk"].get(k, 0.0) / v for k, v in P["tall"].items() if v > 0}}


def team_games(s, t, bases):
    """One row per team-game: observed and expected counts for every rating channel. bases = {season: base_of(...)}."""
    s = s[s.unbl == 1].copy(); t = t.copy()
    s["xg"] = s.xg.fillna(s.groupby(["season", "man"]).xg.transform("mean")).fillna(0.05)
    look = lambda d, tab, col: np.array([bases[int(a)][tab].get(str(b), np.nan) for a, b in zip(d.season, d[col])], float)
    s["q"] = s.xg * s.season.map(lambda a: bases[int(a)]["scale"])
    T5 = t[_reg(t) & (t.man == "5v5")].copy(); S5 = s[_reg(s) & (s.man == "5v5")].copy()
    TM = t[_reg(t)].copy(); SM = s[_reg(s)]
    T5["e_fen"] = T5.dt * look(T5, "b_fen", "sd")
    TGt = T5.groupby(KEY).agg(t5=("dt", "sum"), e_fen=("e_fen", "sum")).reset_index()
    S5["bq"] = look(S5, "b_q", "sd")
    TGs = S5.groupby(KEY).agg(fen=("q", "size"), xg=("q", "sum"), e_xg_given_fen=("bq", "sum")).reset_index()
    TM["e_xg"] = TM.dt * look(TM, "b_xgm", "man")
    pp_t = TM[TM.man.isin(PPM)].groupby(KEY).agg(pp_t=("dt", "sum"), e_ppxg=("e_xg", "sum")).reset_index()
    pk_t = TM[TM.man.isin(PKM)].groupby(KEY).agg(pk_t=("dt", "sum")).reset_index()
    pp_s = SM[SM.man.isin(PPM)].groupby(KEY).agg(ppxg=("q", "sum")).reset_index()
    TM["e_pk"] = TM.dt * look(TM, "b_pk", "sd")
    pen = TM.groupby(KEY).agg(e_pk=("e_pk", "sum")).reset_index()
    fin = s[~s.opull & (s.pb != "ot")].groupby(KEY).agg(g_ne=("goal", "sum"), xg_all=("q", "sum")).reset_index()
    gk = s[~s.opull & s.goalie.notna()].groupby(["game_id", "side", "goalie"]).size().reset_index(name="n").sort_values("n").groupby(["game_id", "side"]).tail(1)
    TG = TGt
    for x in (TGs, pp_t, pk_t, pp_s, pen, fin): TG = TG.merge(x, on=KEY, how="left")
    TG = TG.fillna(0)
    opp = TG[KEY + ["fen", "xg", "e_fen", "e_xg_given_fen", "ppxg", "e_ppxg", "pk_t", "e_pk", "g_ne", "xg_all"]].copy()
    opp["side"] = np.where(opp.side == "home", "away", "home")
    opp = opp.rename(columns={c: "o_" + c for c in opp.columns if c not in ("game_id", "season", "side")}).drop(columns=["o_team", "o_opp"])
    TG = TG.merge(opp, on=["game_id", "season", "side"], how="left")
    gk["side"] = np.where(gk.side == "home", "away", "home")     # shots by one side were faced by the other side's goalie
    TG = TG.merge(gk[["game_id", "side", "goalie"]], on=["game_id", "side"], how="left")    # goalie = own goalie
    return TG.sort_values(["season", "game_id"]).reset_index(drop=True)


def rate(TG):
    """Walk the team-games in order. Returns (pre-game ratings per team-game, team state, goalie state)."""
    dec = 0.5 ** (1 / HL); state, gstate, last, rows = {}, {}, {}, []
    for r in TG.to_dict("records"):
        tm = r["team"]; st = state.setdefault(tm, {c: [0.0, 0.0] for c in CH})
        if last.get(tm) != r["season"]:
            for c in CH: st[c][0] *= CARRY; st[c][1] *= CARRY
            last[tm] = r["season"]
        out = {"game_id": r["game_id"], "side": r["side"], "team": tm, "goalie": r["goalie"]}
        for c, (o, e, k) in CH.items(): out[c] = (k + st[c][0]) / (k + st[c][1])
        gs = gstate.setdefault(r["goalie"], [0.0, 0.0])
        out["gk"] = (GK_K + gs[0]) / (GK_K + gs[1]); rows.append(out)
        for c, (o, e, k) in CH.items():
            st[c][0] = st[c][0] * dec + r[o]; st[c][1] = st[c][1] * dec + r[e]
        gs[0] = gs[0] * dec ** 0.5 + r["o_g_ne"]; gs[1] = gs[1] * dec ** 0.5 + r["o_xg_all"]
    return pd.DataFrame(rows), state, gstate, last


def now_rating(team, season, state, last):
    """A team's rating before its next game."""
    st = state[team]; c0 = CARRY if last.get(team) != season else 1.0
    return {c: (k + c0 * st[c][0]) / (k + c0 * st[c][1]) for c, (o, e, k) in CH.items()}


def goalie_rating(g, gstate):
    gs = gstate.get(g, [0.0, 0.0]); return (GK_K + gs[0]) / (GK_K + gs[1])


def frame(games, rh, ra):
    """games: game_id, season, b2b_h, b2b_a. rh / ra: rating frames (channels + gk) in the same row order."""
    G = games.reset_index(drop=True).copy()
    for s_, r in (("h", rh.reset_index(drop=True)), ("a", ra.reset_index(drop=True))):
        G[f"off5_{s_}"] = r.off_fen * r.off_q; G[f"def5_{s_}"] = r.def_fen * r.def_q
        G[f"pp_{s_}"] = r.pp; G[f"pk_{s_}"] = r.pk; G[f"fin_{s_}"] = r.fin; G[f"gk_{s_}"] = r.gk
        G[f"pt_{s_}"] = r.pen_taken; G[f"pd_{s_}"] = r.pen_drawn
        G[f"rest_{s_}"] = np.where(G[f"b2b_{s_}"].fillna(0) == 1, REST_MULT, 1.0)
    return G


def sim_base(bases):
    """The simulator's tables: 'season|key' strings."""
    out = {"b_fen": {}, "b_q": {}, "b_xgm": {}}
    for s_, b in bases.items():
        for tab in out:
            for k, v in b[tab].items(): out[tab][f"{s_}|{k}"] = v
    return out
