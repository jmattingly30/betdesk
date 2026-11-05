"""Dump the CFB test frames used by model_a_test.py / model_b_test.py (features from the cfb_gap rebuild prelude, same
code cfb_live.py runs) and the 2023-25 drive table for the drive simulator. python3 prep.py (about 40 s)."""
import os, subprocess, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); L = os.path.normpath(f"{HERE}/../rebuild/football/cfb_gap/live")
os.makedirs(f"{HERE}/data", exist_ok=True)
code = ("exec(open('cfb_r6b.py').read().split('\\nE4 = [')[0])\n"
        f"te.copy().to_parquet('{HERE}/data/te_feat.parquet'); fit.copy().to_parquet('{HERE}/data/fit_feat.parquet'); print('frames', te.shape, fit.shape)")
subprocess.run(["python3", "-c", code], cwd=L, check=True)
D = pd.read_parquet(f"{HERE}/../../research/cfb-sim-2026-10/data/drives.parquet")
D[D.season.between(2023, 2025)].to_parquet(f"{HERE}/data/drives_2023_25.parquet"); print("drives 2023-25", (D.season.between(2023, 2025)).sum())
