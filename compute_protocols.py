import os
import glob
import re
import numpy as np
import pandas as pd
from utils import rmse, nasa_score

def compute_all_protocols():
    pred_files = glob.glob("results/preds/*.csv")
    records = []

    for f in sorted(pred_files):
        fname = os.path.basename(f)
        # Pattern: {subset}_{mode}_k{k_override}_f{label_frac}_s{seed}.csv
        # Example: FD004_G_kNone_f1.0_s0.csv or FD004_C_k3_f1.0_s0.csv
        m = re.match(r"(FD\d{3})_([GC])_k([^_]+)_f([^_]+)_s(\d+)\.csv", fname)
        if not m:
            continue
        subset, mode, k_str, f_str, s_str = m.groups()
        k_val = None if k_str == "None" else int(k_str)
        frac_val = float(f_str)
        seed_val = int(s_str)
        method_name = "GRU_global" if mode == "G" else "GRU_cond"

        df = pd.read_csv(f)
        y_true = df["true_RUL"].values.astype(float)
        y_pred = df["pred"].values.astype(float)
        y_true_clipped = np.minimum(y_true, 125.0)

        rmse_standard = rmse(y_true, y_pred)
        rmse_clipped = rmse(y_true_clipped, y_pred)
        nasa_std = nasa_score(y_true, y_pred)
        nasa_clip = nasa_score(y_true_clipped, y_pred)

        records.append({
            "filename": fname,
            "subset": subset,
            "mode": mode,
            "method": method_name,
            "k": k_val,
            "label_frac": frac_val,
            "seed": seed_val,
            "rmse_standard": rmse_standard,
            "rmse_clipped": rmse_clipped,
            "nasa_standard": nasa_std,
            "nasa_clipped": nasa_clip
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv("results/protocol_rmse.csv", index=False)
    print(f"Computed protocol comparisons for {len(out_df)} prediction files. Saved to results/protocol_rmse.csv.")
    return out_df

if __name__ == "__main__":
    compute_all_protocols()

