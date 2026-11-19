import numpy as np
from train import train_gru

for mode in ["G", "C"]:
    r, s, el, ep, d = train_gru("FD004", mode=mode, seed=0, max_epochs=25,
                                patience=6, batch_size=512, lr=2e-3,
                                verbose=True, return_details=True)
    print(f"\n=== FD004 mode {mode} ===")
    print(f"test RMSE {r:.2f} | NASA {s:.1f} | epochs {ep} | {d['sec_per_epoch']:.1f}s/epoch")
    print(f"train RMSE {d['train_rmse']:.2f} | val RMSE {d['val_rmse']:.2f} | trivial RMSE {d['trivial_rmse']:.2f}")
    print(f"preds min/max/mean/std: {d['pred_min']:.1f} / {d['pred_max']:.1f} / {d['pred_mean']:.1f} / {d['pred_std']:.1f}")
    print(f"true  min/max/mean/std: {d['y_test'].min():.1f} / {d['y_test'].max():.1f} / {d['y_test'].mean():.1f} / {d['y_test'].std():.1f}\n", flush=True)