"""NFL Model B blend (J 2026-11-06: every NFL chance must carry our model; show the best blend of our model + the line).
blended = line + w x (Model A - line). Fit w by least squares on margins and totals, leave-one-season-out 2018-2026 against the
nflverse closing line; check the 2026 openers (open_test frame). Also scores the chances: cover log loss of the blend vs line alone.
python3 blend_test.py -> blend_test.json (live.py reads w_spread / w_total)"""
import os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
S = pd.read_parquet(f"{HERE}/../rebuild/football/data/stack_nfl.parquet")
S = S[(S.season >= 2018) & S.line_margin.notna() & S.line_total.notna() & (S.week > 1)].copy()
S["g_m"] = (S.pred_h - S.pred_a) - S.line_margin; S["r_m"] = (S.act_h - S.act_a) - S.line_margin
S["g_t"] = (S.pred_h + S.pred_a) - S.line_total; S["r_t"] = (S.act_h + S.act_a) - S.line_total
out = {}
for k in ("m", "t"):
    w_all = float((S[f"g_{k}"] * S[f"r_{k}"]).sum() / (S[f"g_{k}"] ** 2).sum())
    loso = []
    for s in sorted(S.season.unique()):
        tr, te = S[S.season != s], S[S.season == s]
        w = (tr[f"g_{k}"] * tr[f"r_{k}"]).sum() / (tr[f"g_{k}"] ** 2).sum()
        loso.append({"season": int(s), "w": round(float(w), 3), "mse_line": round(float((te[f"r_{k}"] ** 2).mean()), 2), "mse_blend": round(float(((te[f"r_{k}"] - w * te[f"g_{k}"]) ** 2).mean()), 2)})
    L = pd.DataFrame(loso); out[k] = {"w_all": round(w_all, 3), "seasons_blend_better": int((L.mse_blend < L.mse_line).sum()), "n_seasons": len(L),
                                      "mse_line": round(float(L.mse_line.mean()), 3), "mse_blend": round(float(L.mse_blend.mean()), 3), "by_season": loso}
    print(k, out[k]["w_all"], out[k]["seasons_blend_better"], "/", len(L), out[k]["mse_line"], out[k]["mse_blend"])
json.dump(out, open(f"{HERE}/blend_test.json", "w"), indent=1)
