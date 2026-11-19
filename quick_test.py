import time
from train import train_gru

if __name__ == "__main__":
    t0 = time.time()
    print("Running quick_test.py on FD004 (Mode C, k_override=3, label_frac=0.25, 2 epochs)...", flush=True)
    r, s, el, ep = train_gru(
        subset="FD004",
        mode="C",
        seed=0,
        max_epochs=2,
        patience=2,
        batch_size=512,
        lr=2e-3,
        k_override=3,
        label_frac=0.25,
        verbose=True
    )
    print(f"Passed in {el:.2f}s! RMSE={r:.2f}, NASA={s:.1f}, Epochs={ep}", flush=True)

