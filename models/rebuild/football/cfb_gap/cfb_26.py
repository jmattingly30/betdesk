import pandas as pd, numpy as np
import cfb_lib as L
A = pd.read_parquet("cfb_te_all26.parquet"); t26 = A[A.season == 2026].copy()
print("2026 games with v3 + rebuild:", len(t26), "weeks", t26.wk.min(), "-", t26.wk.max())
pe = L.run(['model_margin'] + L.P + L.RT + L.D, Tt=t26); pl = L.run(['model_margin'] + L.P, Tt=t26)
w = t26.wk.values <= 3; v4 = np.where(w, pe, pl); v3 = t26.model_margin.values
e3 = np.abs(v3 - t26.am.values); e4 = np.abs(v4 - t26.am.values); d = e4 - e3
print("2026 margin MAE v3", e3.mean().round(3), "v4", e4.mean().round(3), "diff", d.mean().round(3), "t", (d.mean() / (d.std() / np.sqrt(len(d)))).round(1))
print("  wk1-3", e3[w].mean().round(3), e4[w].mean().round(3), "| wk4+", e3[~w].mean().round(3), e4[~w].mean().round(3))
