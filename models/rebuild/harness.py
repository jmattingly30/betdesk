"""Shared grading harness for the score-projection rebuild (all sports).

Every model hands in one table, one row per game:
  sport, game_id, season, date, home, away,
  pred_h, pred_a            projected points for each team (means)
  sd_margin, sd_total       spread of the projected distribution (optional; sims give these)
  rho                       correlation of the two team scores (optional)
  p_home                    win chance (optional; else from the normal approximation)
plus the actual result: act_h, act_a.
Optional market yardstick: line_margin (expected home margin implied by the closing spread),
line_total. The close is a yardstick only, never an input.

Main calls
  score(df)                 -> dict of metrics
  by(df, col)               -> metrics per segment (season, month, week...)
  compare(a, b, loss)       -> paired test of model b vs model a on the same games
  bh(pvals, q)              -> Benjamini-Hochberg decisions (false-discovery control)
  log_result(...)           -> appends a row to ledger.csv ("earns its place" record)
"""
import math, os, datetime as dt
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "ledger.csv")

def _norm_crps(mu, sd, y):
    z = (y - mu) / sd
    return sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))

def prepare(df):
    d = df.copy()
    d["margin_hat"] = d.pred_h - d.pred_a
    d["total_hat"] = d.pred_h + d.pred_a
    d["margin"] = d.act_h - d.act_a
    d["total"] = d.act_h + d.act_a
    if "p_home" not in d or d["p_home"].isna().all():
        if "sd_margin" in d:
            d["p_home"] = stats.norm.cdf(d.margin_hat / d.sd_margin)
    return d

def score(df):
    d = prepare(df)
    e_h, e_a = d.pred_h - d.act_h, d.pred_a - d.act_a
    team_err = np.concatenate([e_h, e_a])
    em, et = d.margin_hat - d.margin, d.total_hat - d.total
    out = {
        "n": len(d),
        "team_mae": np.mean(np.abs(team_err)), "team_rmse": np.sqrt(np.mean(team_err**2)),
        "team_bias": np.mean(team_err),
        "margin_mae": np.mean(np.abs(em)), "margin_rmse": np.sqrt(np.mean(em**2)),
        "total_mae": np.mean(np.abs(et)), "total_rmse": np.sqrt(np.mean(et**2)), "total_bias": np.mean(et),
    }
    if "sd_margin" in d and d.sd_margin.notna().all():
        out["margin_crps"] = np.mean(_norm_crps(d.margin_hat, d.sd_margin, d.margin))
        z = np.abs(em) / d.sd_margin
        out["margin_cover80"] = np.mean(z < 1.2816)   # should be ~0.80
    if "sd_total" in d and d.sd_total.notna().all():
        out["total_crps"] = np.mean(_norm_crps(d.total_hat, d.sd_total, d.total))
        out["total_cover80"] = np.mean(np.abs(et) / d.sd_total < 1.2816)
    if "p_home" in d and d.p_home.notna().any():
        m = d.margin != 0
        p = d.p_home[m].clip(1e-4, 1 - 1e-4); y = (d.margin[m] > 0).astype(float)
        out["win_logloss"] = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        out["win_brier"] = np.mean((p - y) ** 2)
    if "line_margin" in d and d.line_margin.notna().any():
        m = d.line_margin.notna()
        out["n_line"] = int(m.sum())
        out["margin_mae_model_on_lined"] = np.mean(np.abs(em[m]))
        out["margin_mae_close"] = np.mean(np.abs(d.line_margin[m] - d.margin[m]))
        out["margin_rmse_model_on_lined"] = np.sqrt(np.mean(em[m] ** 2))
        out["margin_rmse_close"] = np.sqrt(np.mean((d.line_margin[m] - d.margin[m]) ** 2))
    if "line_total" in d and d.line_total.notna().any():
        m = d.line_total.notna()
        out["total_mae_model_on_lined"] = np.mean(np.abs(et[m]))
        out["total_mae_close"] = np.mean(np.abs(d.line_total[m] - d.total[m]))
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}

def by(df, col):
    return pd.DataFrame({k: score(g) for k, g in df.groupby(col)}).T

LOSSES = {
    "team_abs": lambda d: (np.abs(d.pred_h - d.act_h) + np.abs(d.pred_a - d.act_a)) / 2,
    "margin_abs": lambda d: np.abs(d.margin_hat - d.margin),
    "total_abs": lambda d: np.abs(d.total_hat - d.total),
    "margin_sq": lambda d: (d.margin_hat - d.margin) ** 2,
    "total_sq": lambda d: (d.total_hat - d.total) ** 2,
    "margin_crps": lambda d: _norm_crps(d.margin_hat, d.sd_margin, d.margin),
    "total_crps": lambda d: _norm_crps(d.total_hat, d.sd_total, d.total),
    "win_ll": lambda d: -np.where(d.margin > 0, np.log(d.p_home.clip(1e-4, 1 - 1e-4)),
                                  np.log(1 - d.p_home.clip(1e-4, 1 - 1e-4))) * (d.margin != 0),
}

def compare(a, b, loss="team_abs", cluster="date"):
    """Paired test of b vs a on games both graded. Negative delta = b is better.
    Standard error clustered by date (games on the same day share news/conditions)."""
    da, db = prepare(a), prepare(b)
    m = da.merge(db, on="game_id", suffixes=("_a", "_b"))
    la = LOSSES[loss](da.set_index("game_id").loc[m.game_id]).values
    lb = LOSSES[loss](db.set_index("game_id").loc[m.game_id]).values
    diff = lb - la
    cl = pd.to_datetime(m[f"{cluster}_a"]).dt.date.values if f"{cluster}_a" in m else np.arange(len(m))
    g = pd.Series(diff).groupby(cl).sum()
    n = len(diff); G = len(g)
    se = math.sqrt((g ** 2).sum() * G / max(G - 1, 1)) / n
    delta = diff.mean(); t = delta / se if se > 0 else 0.0
    p = 2 * stats.t.sf(abs(t), df=max(G - 1, 1))
    return {"loss": loss, "n": n, "a": round(float(la.mean()), 4), "b": round(float(lb.mean()), 4),
            "delta": round(float(delta), 4), "se": round(float(se), 4), "t": round(float(t), 2), "p": float(p)}

def bh(pvals, q=0.10):
    p = np.asarray(pvals, float); n = len(p); order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    keep = np.zeros(n, bool); keep[order[:k]] = True
    return keep

def log_result(sport, stage, module, baseline, result, decision, note=""):
    row = {"when": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M"), "sport": sport, "stage": stage,
           "module": module, "baseline": baseline, **{k: result.get(k) for k in ("loss", "n", "a", "b", "delta", "t", "p")},
           "decision": decision, "note": note}
    pd.DataFrame([row]).to_csv(LEDGER, mode="a", header=not os.path.exists(LEDGER), index=False)
    return row
