"""Missing-star test for CFB Model A (J 2026-11-05: Malachi Toney / Julian Sayin out; elite absences must be a MODEL INPUT).
Stars come from play-by-play usage in the team's earlier games this season (data/usage_games.parquet):
  QB   = led the team in pass attempts in 2+ of its last 3 games; elite = 225+ pass yds/game in those games
  SKILL= top scrimmage-yards player (rush + rec) in 2+ of last 4; elite = 90+ scrimmage yds/game (props rule: 75 rush / 60 rec)
Absent = no rush, target or pass attempt in this game. First-missed game vs later missed games are split (v3's walk-forward
ratings already see the later ones). Effect = team score minus Model A (v4) team score; market = closing implied team total
minus Model A; the input is fit on 2022-23 and checked on 2024-25 (team MAE vs v4), then on all 4 seasons for live.
python3 absence_test.py -> absence_test.json, data/absences_hist.parquet"""
import json, numpy as np, pandas as pd
from lib import *
U = pd.read_parquet(f"{HERE}/data/usage_games.parquet"); U = U[U.stype.astype(str) != '3']
fit = pd.read_parquet(f"{HERE}/data/fit_feat.parquet"); te = pd.read_parquet(f"{HERE}/data/te_feat.parquet")
te["m4"], te["t4"] = v4(fit, te); te["h4"], te["a4"] = (te.t4 + te.m4) / 2, (te.t4 - te.m4) / 2
te["ih"], te["ia"] = (te.total_close + te.line_close) / 2, (te.total_close - te.line_close) / 2

def stars(U):
    """for each (season, team, game in order): list of stars who are absent, with kind / elite / missed-streak"""
    U = U.copy(); U["scr"] = U.rush_yds + U.rec_yds
    order = U.groupby(["season", "team", "game_id"]).week.first().reset_index().sort_values(["season", "team", "week"])
    rows = []
    for (s, t), g in order.groupby(["season", "team"]):
        gids = list(g.game_id); P = U[(U.season == s) & (U.team == t)].set_index("game_id")
        hist = []   # per earlier game: (qb, qb_yds, top_scr, scr_yds, players set)
        for gid in gids:
            cur = P.loc[[gid]] if gid in P.index else P.iloc[0:0]; present = set(cur.player)
            if len(hist) >= 3:
                qbs = [h[0] for h in hist[-3:] if h[0]]; top = [h[2] for h in hist[-4:] if h[2]]
                for kind, cand, last in (("qb", qbs, hist[-3:]), ("skill", top, hist[-4:])):
                    if not cand: continue
                    p = max(set(cand), key=cand.count)
                    if cand.count(p) < 2 or p in present: continue
                    if kind == "qb": avg = np.mean([h[1] for h in last if h[0] == p]); elite = avg >= 225
                    else: avg = np.mean([h[3] for h in last if h[2] == p]); elite = avg >= 90
                    streak = 0
                    for h in reversed(hist):
                        if p in h[4]: break
                        streak += 1
                    rows.append(dict(season=s, team=int(t), game_id=gid, player=p, kind=kind, avg=round(float(avg), 1), elite=bool(elite), streak=streak))
            if len(cur):
                q = cur.sort_values("pass_att").iloc[-1]; k = cur.sort_values("scr").iloc[-1]
                hist.append((q.player if q.pass_att >= 8 else None, float(q.pass_yds), k.player, float(k.scr), present))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    A = stars(U); A.to_parquet(f"{HERE}/data/absences_hist.parquet"); print("absences", len(A), A.groupby(["kind", "elite"]).size().to_dict())
    rows = []
    for side, tid, act, pred, imp in (("h", "home_id", "act_h", "h4", "ih"), ("a", "away_id", "act_a", "a4", "ia")):
        x = te[["game_id", "season", "wk", tid, act, pred, imp]].set_axis(["game_id", "season", "wk", "team", "act", "pred", "imp"], axis=1); x["side"] = side; rows.append(x)
    T = pd.concat(rows, ignore_index=True); T["game_id"] = T.game_id.astype(str); T["team"] = T.team.astype(int)
    key = lambda d, f: d.set_index(["game_id", "team"]).index
    for kind in ("qb", "skill"):
        for el in (True, False):
            for first in (True, False):
                sub = A[(A.kind == kind) & (A.elite == el) & ((A.streak == 0) == first)]
                T[f"{kind}_{'e' if el else 'n'}_{'1' if first else 'l'}"] = key(T, 0).isin(sub.set_index(["game_id", "team"]).index).astype(float)
    T["r"] = T.act - T.pred; T["mk"] = T.imp - T.pred
    cols = [c for c in T.columns if c[:3] in ("qb_", "ski")]
    res = {"team_games": len(T), "by_group": {}}
    for c in cols:
        m = T[c] == 1
        res["by_group"][c] = {"n": int(m.sum()), "team_minus_A": round(float(T.r[m].mean()), 2) if m.any() else None,
                              "close_minus_A": round(float(T.mk[m].mean()), 2) if m.any() else None, "se": round(float(T.r[m].std() / np.sqrt(max(m.sum(), 1))), 2) if m.sum() > 1 else None}
    # input: one coefficient per (kind x elite), first + later missed games pooled, then a later-game discount if it helps
    X_cols = {"qb_e": ["qb_e_1", "qb_e_l"], "qb_n": ["qb_n_1", "qb_n_l"], "sk_e": ["skill_e_1", "skill_e_l"], "sk_n": ["skill_n_1", "skill_n_l"]}
    for k, cs in X_cols.items(): T[k] = T[cs].sum(1)
    feats = list(X_cols)
    def fitc(F, cols): X = F[cols].values; return np.linalg.solve(X.T @ X + 5 * np.eye(len(cols)), X.T @ F.r.values)
    for nm, cols in (("pooled", feats), ("first_only", ["qb_e_1", "qb_n_1", "skill_e_1", "skill_n_1"])):
        tr, ts = T[T.season <= 2023], T[T.season >= 2024]; b = fitc(tr, cols); p = ts[cols].values @ b
        e0, e1 = np.abs(ts.r), np.abs(ts.r - p); d = e1 - e0; aff = ts[cols].sum(1) > 0
        res[f"oos_{nm}"] = {"coef_fit_22_23": dict(zip(cols, np.round(b, 2).tolist())), "team_MAE_A": round(float(e0.mean()), 4), "team_MAE_A_plus": round(float(e1.mean()), 4),
                            "diff": round(float(d.mean()), 4), "t": round(float(d.mean() / (d.std() / np.sqrt(len(d)))), 2),
                            "affected_n": int(aff.sum()), "affected_diff": round(float(d[aff].mean()), 3) if aff.any() else None,
                            "vs_close_affected": {"A_plus_MAE": round(float(np.abs(ts.act - ts.pred - p)[aff].mean()), 3), "A_MAE": round(float(e0[aff].mean()), 3), "close_MAE": round(float(np.abs(ts.act - ts.imp)[aff].mean()), 3)}}
        print(nm, res[f"oos_{nm}"])
    res["live_coef"] = dict(zip(feats, np.round(fitc(T, feats), 2).tolist()))
    print(json.dumps(res["by_group"], indent=0)); print("live", res["live_coef"])
    json.dump(res, open(f"{HERE}/absence_test.json", "w"), indent=1)
