import numpy as np
import pandas as pd

def nasa_score(y_true, y_pred):
    """
    Computes the NASA asymmetric scoring metric:
    d = y_pred - y_true
    score = sum(exp(-d/13) - 1 for d < 0, exp(d/10) - 1 for d >= 0)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))

def rmse(y_true, y_pred):
    """
    Computes Root Mean Squared Error.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def make_last_window(df, features, window=30):
    """
    For every ENGINE in df, returns one sample made of its last `window` cycles.
    If an engine has fewer than `window` cycles, pad by repeating its first row at the start.
    Returns:
        X: 2D numpy array of shape (num_engines, window * len(features))
        unit_ids: list of engine unit numbers in the same order
    """
    X_list = []
    unit_ids = []
    
    # Sort by unit_number and time_in_cycles
    df_sorted = df.sort_values(by=['unit_number', 'time_in_cycles'])
    
    for unit, group in df_sorted.groupby('unit_number', sort=True):
        vals = group[features].values
        n = len(vals)
        if n >= window:
            win = vals[-window:]
        else:
            pad_len = window - n
            pad = np.repeat(vals[0:1], pad_len, axis=0)
            win = np.vstack([pad, vals])
        
        X_list.append(win.flatten())
        unit_ids.append(unit)
        
    return np.array(X_list, dtype=np.float64), np.array(unit_ids)

def make_train_windows(df, features, window=30, stride=5):
    """
    Extracts sliding windows of length `window` with step `stride` from train engines.
    Target RUL_clipped is taken at the window's last cycle.
    Returns:
        X: 2D numpy array of shape (num_windows, window * len(features))
        y: 1D numpy array of shape (num_windows,)
    """
    X_list = []
    y_list = []
    
    df_sorted = df.sort_values(by=['unit_number', 'time_in_cycles'])
    
    for unit, group in df_sorted.groupby('unit_number', sort=True):
        feat_vals = group[features].values
        rul_vals = group['RUL_clipped'].values
        n = len(feat_vals)
        
        if n >= window:
            for start_idx in range(0, n - window + 1, stride):
                end_idx = start_idx + window
                win = feat_vals[start_idx:end_idx].flatten()
                target = rul_vals[end_idx - 1]
                X_list.append(win)
                y_list.append(target)
        else:
            # Pad if shorter than window
            pad_len = window - n
            pad = np.repeat(feat_vals[0:1], pad_len, axis=0)
            win = np.vstack([pad, feat_vals]).flatten()
            target = rul_vals[-1]
            X_list.append(win)
            y_list.append(target)
            
    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)

if __name__ == "__main__":
    print("=" * 60)
    print("TASK 2: TESTING UTILITY FUNCTIONS (utils.py)")
    print("=" * 60)
    
    # 1. Test nasa_score and rmse
    y_true = np.array([100.0, 50.0, 20.0])
    # Case 1: perfect prediction
    print("\n1. Testing metrics on perfect prediction (d=0):")
    print(f"  RMSE: {rmse(y_true, y_true):.4f} (expected: 0.0000)")
    print(f"  NASA Score: {nasa_score(y_true, y_true):.4f} (expected: 0.0000)")
    
    # Case 2: early vs late prediction
    # d = -5 (early): exp(5/13) - 1 = 0.469
    # d = +5 (late):  exp(5/10) - 1 = 0.6487
    y_early = np.array([95.0, 45.0, 15.0])
    y_late  = np.array([105.0, 55.0, 25.0])
    print("\n2. Testing asymmetric penalty (early vs late):")
    print(f"  Early prediction (d = -5 each): NASA Score = {nasa_score(y_true, y_early):.4f}")
    print(f"  Late prediction  (d = +5 each): NASA Score = {nasa_score(y_true, y_late):.4f}")
    assert nasa_score(y_true, y_late) > nasa_score(y_true, y_early), "Late penalty should be strictly larger than early penalty!"

    # 3. Test make_last_window and make_train_windows on synthetic data
    print("\n3. Testing window generators:")
    dummy_df = pd.DataFrame({
        'unit_number': [1]*35 + [2]*20, # Engine 1 has 35 cycles (>30), Engine 2 has 20 cycles (<30)
        'time_in_cycles': list(range(1, 36)) + list(range(1, 21)),
        'feat_a': [10.0 + i for i in range(35)] + [100.0 + i for i in range(20)],
        'feat_b': [20.0 + i for i in range(35)] + [200.0 + i for i in range(20)],
        'RUL_clipped': [125.0 - i for i in range(35)] + [50.0 - i for i in range(20)]
    })
    
    feats = ['feat_a', 'feat_b']
    X_last, units = make_last_window(dummy_df, feats, window=30)
    print(f"  make_last_window output shape: {X_last.shape} (expected: (2, 60))")
    print(f"  Engine units returned: {units} (expected: [1, 2])")
    assert X_last.shape == (2, 60), f"Expected shape (2, 60), got {X_last.shape}"
    assert np.array_equal(units, [1, 2]), "Unit IDs mismatch"
    
    # Check padding on Engine 2 (first row repeated 10 times at the beginning)
    # feats at first cycle of engine 2: [100.0, 200.0]
    eng2_first_cycle = X_last[1, :2]
    eng2_tenth_cycle = X_last[1, 18:20]
    print(f"  Padded engine 2: cycle 1={eng2_first_cycle}, cycle 10={eng2_tenth_cycle} (both should be [100.0, 200.0])")
    assert np.array_equal(eng2_first_cycle, [100.0, 200.0])
    assert np.array_equal(eng2_tenth_cycle, [100.0, 200.0])

    X_train, y_train = make_train_windows(dummy_df, feats, window=30, stride=5)
    print(f"  make_train_windows output: X shape = {X_train.shape}, y shape = {y_train.shape}")
    # Engine 1 has 35 cycles: start indices 0, 5 -> 2 windows. Engine 2 has 20 cycles: 1 padded window -> total 3 windows.
    print(f"  Total train windows: {len(X_train)} (expected: 3)")
    assert len(X_train) == 3, f"Expected 3 windows, got {len(X_train)}"
    assert len(y_train) == 3, f"Expected 3 targets, got {len(y_train)}"
    
    print("\nAll utility function tests passed successfully!")
    print("=" * 60)

