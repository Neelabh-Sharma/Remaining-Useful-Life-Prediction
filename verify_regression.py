import time
from train import train_gru

print("Starting regression verification on FD004 GRU_cond seed 0 (max_epochs=40, patience=8, batch_size=512, lr=2e-3)...", flush=True)
t0 = time.time()
te_rmse, te_nasa, elapsed, ep = train_gru(
    subset="FD004",
    mode="C",
    seed=0,
    max_epochs=40,
    patience=8,
    batch_size=512,
    lr=2e-3,
    verbose=True,
    save_preds=False
)
print(f"Result: Test RMSE = {te_rmse:.4f} (target: 26.33 +/- 0.05), NASA = {te_nasa:.1f}, Epochs = {ep}, Elapsed = {elapsed:.1f}s", flush=True)
diff = abs(te_rmse - 26.33)
print(f"Absolute diff from 26.33: {diff:.4f}", flush=True)
assert diff <= 0.05, f"Regression check failed: {te_rmse:.4f} is not within 0.05 of 26.33!"
print("REGRESSION CHECK PASSED!", flush=True)
