import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/stage2_tuning.csv")
fd4 = df[df["subset"] == "FD004"]

print("================================================================================")
print("STAGE 2 TUNING RESULTS (FD004 Seeds 0 and 1)")
print("================================================================================")

summary_rows = []
for cfg in ["A", "B", "C", "D"]:
    sub = fd4[fd4["config"] == cfg]
    val_mean = sub["val_rmse"].mean()
    val_g = sub[sub["method"] == "GRU_global"]["val_rmse"].mean()
    val_c = sub[sub["method"] == "GRU_cond"]["val_rmse"].mean()
    test_mean = sub["test_rmse"].mean()
    test_g = sub[sub["method"] == "GRU_global"]["test_rmse"].mean()
    test_c = sub[sub["method"] == "GRU_cond"]["test_rmse"].mean()
    time_mean = sub["elapsed_sec"].mean()
    
    summary_rows.append({
        "Config": cfg,
        "Val RMSE (Overall)": val_mean,
        "Val RMSE (Global)": val_g,
        "Val RMSE (Cond)": val_c,
        "Test RMSE (Overall)": test_mean,
        "Test RMSE (Global)": test_g,
        "Test RMSE (Cond)": test_c,
        "Mean Time (s)": time_mean
    })

sum_df = pd.DataFrame(summary_rows)
base_val = sum_df.loc[sum_df["Config"] == "A", "Val RMSE (Overall)"].values[0]
sum_df["Val vs A (%)"] = (sum_df["Val RMSE (Overall)"] - base_val) / base_val * 100
print(sum_df.to_string(index=False))

print("\n================================================================================")
print("PER-RUN DETAILS (FD004)")
print("================================================================================")
for cfg in ["A", "B", "C", "D"]:
    print(f"\n--- CONFIG {cfg} ---")
    sub = fd4[fd4["config"] == cfg]
    for _, r in sub.iterrows():
        print(f"  {r['method']} (seed {r['seed']}): Val RMSE = {r['val_rmse']:.2f} | Test RMSE = {r['test_rmse']:.2f} | NASA = {r['nasa_score']:.1f} | Epochs = {r['actual_epochs']} | Time = {r['elapsed_sec']:.1f}s")

print("\n================================================================================")
print("FD003 EARLY STOPPING DIAGNOSTIC")
print("================================================================================")
fd3 = df[df["subset"] == "FD003"]
for _, r in fd3.iterrows():
    print(f"  Config {r['config']} Seed {r['seed']}: Val RMSE = {r['val_rmse']:.2f} | Test RMSE = {r['test_rmse']:.2f} | NASA = {r['nasa_score']:.1f} | Epochs = {r['actual_epochs']} | Time = {r['elapsed_sec']:.1f}s")

# Let's inspect FD003 validation curves from HISTORY_JSON
with open("results/stage2_history.json", "r") as f:
    hist = json.load(f)

# Plot validation curves for FD003
fig, ax = plt.subplots(figsize=(8, 4.5))
for key in ["A_FD003_GRU_cond_s0", "A_FD003_GRU_cond_s1", "A_FD003_GRU_cond_s2", "D_FD003_GRU_cond_s0"]:
    if key in hist:
        vl = np.sqrt(np.array(hist[key]["val_loss"])) * 125.0
        epochs = np.arange(1, len(vl) + 1)
        ax.plot(epochs, vl, label=f"{key} (best epoch: {np.argmin(vl)+1})")

ax.set_title("FD003 Validation Loss Curves: Premature Early Stopping vs Extended Training")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation RMSE (Cycles)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()
plt.tight_layout()
fig.savefig("figures/fd003_validation_curves.png", dpi=300)
fig.savefig("figures/fd003_validation_curves.pdf")
plt.close(fig)
print("\nSaved figures/fd003_validation_curves.png and .pdf")
