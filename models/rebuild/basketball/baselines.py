"""Today's CBB and NBA models in harness format (the bar every rebuild must clear)."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import harness as H
M = "/mnt/project-files/sports-analytics/models"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"); os.makedirs(OUT, exist_ok=True)

def cbb():
    p = pd.read_parquet(f"{M}/cbb/data/cbb_predictions.parquet")
    d = pd.DataFrame({"sport": "CBB", "game_id": p.game_id, "season": p.season, "date": pd.to_datetime(p.date),
        "home": p.home, "away": p.away, "neutral": p.neutral, "pred_h": p.pts_h_hat, "pred_a": p.pts_a_hat,
        "act_h": p.home_score, "act_a": p.away_score, "sd_margin": p.sigma, "p_home": p.p_win})
    # verified closes only (consensus and TeamRankings agree); line = expected home margin
    d["line_margin"] = np.where(p.verified == True, p.line, np.nan)
    d["line_total"] = p.total_line
    d["sd_total"] = np.nan
    return d

def nba():
    p = pd.read_parquet(f"{M}/nba-team/data/predictions.parquet")
    cfg = pd.read_json(f"{M}/nba-team/model_config.json", typ="series")
    sig = float(cfg.get("sigma", 13.2)) if "sigma" in cfg else 13.2
    d = pd.DataFrame({"sport": "NBA", "game_id": p.game_id, "season": p.season, "date": pd.to_datetime(p.date),
        "home": p.home, "away": p.away, "pred_h": p.pts_h_hat, "pred_a": p.pts_a_hat,
        "act_h": p.home_score, "act_a": p.away_score, "p_home": p.p_win, "sd_margin": sig,
        "line_margin": p.close, "line_total": p.total_close, "playoff": p.playoff})
    return d

def nba_known():
    """Same model priced with only the absences known at the open (no hindsight about who sat)."""
    p = pd.read_parquet(f"{M}/nba-team/data/predictions.parquet")
    d = nba()
    d["pred_h"] = (p.total_known + p.margin_known).values / 2; d["pred_a"] = (p.total_known - p.margin_known).values / 2
    from scipy.stats import norm
    d["p_home"] = norm.cdf(p.margin_known.values / d.sd_margin.values)
    return d

if __name__ == "__main__":
    for name, f in (("cbb", cbb), ("nba", nba), ("nba_known", nba_known)):
        d = f(); d.to_parquet(f"{OUT}/baseline_{name}.parquet")
        print(name.upper(), "corr(line, margin) =", round(d.line_margin.corr(d.act_h - d.act_a), 3))
        print(pd.Series(H.score(d)).to_string())
        print(H.by(d[d.season >= d.season.max() - 4], "season")[["n", "team_mae", "margin_mae", "total_mae", "team_bias"]])
        print()
