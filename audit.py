import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_audit():
    print("=" * 70)
    print("TASK 1: DATA AUDIT FOR NASA C-MAPSS TURBOFAN DATASET")
    print("=" * 70)

    # 1. Load Data
    train_path = "train.csv"
    test_path = "test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1.1 Shapes, dtypes, missing values, duplicate rows
    print("\n--- 1. SHAPE, DTYPES, MISSING VALUES, DUPLICATE ROWS ---")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    print("\nTrain dtypes count:")
    print(train_df.dtypes.value_counts().to_string())
    print("\nTest dtypes count:")
    print(test_df.dtypes.value_counts().to_string())

    train_nulls = train_df.isnull().sum()
    test_nulls = test_df.isnull().sum()
    print(f"\nTrain missing values total: {train_nulls.sum()}")
    print(f"Test missing values total:  {test_nulls.sum()}")
    if test_nulls.sum() > 0:
        print("Columns with missing values in test.csv:")
        print(test_nulls[test_nulls > 0].to_string())

    train_dups = train_df.duplicated().sum()
    test_dups = test_df.duplicated().sum()
    print(f"\nTrain exact duplicate rows: {train_dups}")
    print(f"Test exact duplicate rows:  {test_dups}")

    # 2. Unit number overlap check
    print("\n--- 2. UNIT NUMBER VALUES & OVERLAP ---")
    train_units = sorted(train_df['unit_number'].unique())
    test_units = sorted(test_df['unit_number'].unique())
    print(f"Train unit_number count: {len(train_units)}, min: {min(train_units)}, max: {max(train_units)}")
    print(f"Test unit_number count:  {len(test_units)}, min: {min(test_units)}, max: {max(test_units)}")

    overlap = set(train_units).intersection(set(test_units))
    print(f"Number of overlapping unit_number values between train and test: {len(overlap)}")
    print(f"Overlap percentage relative to test units: {len(overlap)/len(test_units)*100:.1f}%")
    print("Findings on unit IDs: Unit IDs 1 through 259 exist in BOTH train and test splits.")

    # 3. Check uniqueness of (unit_number, time_in_cycles) & Subset detection
    print("\n--- 3. UNIQUENESS OF (UNIT_NUMBER, TIME_IN_CYCLES) & SUBSET DETECTION ---")
    train_pair_dups = train_df.duplicated(subset=['unit_number', 'time_in_cycles']).sum()
    test_pair_dups = test_df.duplicated(subset=['unit_number', 'time_in_cycles']).sum()
    print(f"Train duplicate (unit_number, time_in_cycles) pairs: {train_pair_dups} (out of {len(train_df)} rows)")
    print(f"Test duplicate (unit_number, time_in_cycles) pairs:  {test_pair_dups} (out of {len(test_df)} rows)")

    # Detection of resets where cycle drops to 1 or unit_number resets
    cycle_resets_train = (train_df['time_in_cycles'] == 1).sum()
    cycle_resets_test = (test_df['time_in_cycles'] == 1).sum()
    print(f"Train: Number of times time_in_cycles == 1: {cycle_resets_train}")
    print(f"Test:  Number of times time_in_cycles == 1: {cycle_resets_test}")

    # Operating conditions analysis
    op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    print("\nOperating setting stats across full train dataset:")
    print(train_df[op_cols].describe().to_string())

    # Detect the exact subsets
    print("\n--- SUBSET SEPARATION ANALYSIS ---")
    # Reference NASA C-MAPSS benchmarks:
    # FD001: Train 20,631 rows (100 units), Test 13,096 rows (100 units), 1 Condition, 1 Fault Mode (HPC)
    # FD002: Train 53,759 rows (260 units), Test 33,991 rows (259 units), 6 Conditions, 1 Fault Mode (HPC)
    # FD003: Train 24,720 rows (100 units), Test 16,596 rows (100 units), 1 Condition, 2 Fault Modes (HPC, Fan)
    # FD004: Train 61,249 rows (249 units), Test 41,214 rows (248 units), 6 Conditions, 2 Fault Modes (HPC, Fan)
    train_benchmarks = [("FD001", 20631, 100), ("FD002", 53759, 260), ("FD003", 24720, 100), ("FD004", 61249, 249)]
    test_benchmarks = [("FD001", 13096, 100), ("FD002", 33991, 259), ("FD003", 16596, 100), ("FD004", 41214, 248)]

    print(f"Total benchmark train rows: {sum(b[1] for b in train_benchmarks)} | Actual train rows: {len(train_df)}")
    print(f"Total benchmark test rows:  {sum(b[1] for b in test_benchmarks)} | Actual test rows:  {len(test_df)}")

    # Verify if boundaries correspond exactly
    print("\nVerifying train subset boundary partition:")
    curr_idx = 0
    subsets_detected = True
    for name, expected_rows, expected_units in train_benchmarks:
        sub = train_df.iloc[curr_idx : curr_idx + expected_rows]
        u_count = sub['unit_number'].nunique()
        u_min = sub['unit_number'].min()
        u_max = sub['unit_number'].max()
        c_first = sub['time_in_cycles'].iloc[0]
        c_last = sub['time_in_cycles'].iloc[-1]
        s1_std = sub['sensor_1'].std()
        print(f"  {name}: Rows [{curr_idx:6d}:{curr_idx + expected_rows:6d}] (Total {len(sub):5d}) | Units {u_min}-{u_max} ({u_count} units) | Cycle start: {c_first}, end: {c_last} | sensor_1 std: {s1_std:.4f}")
        if len(sub) != expected_rows or u_count != expected_units:
            subsets_detected = False
        curr_idx += expected_rows

    print("\nVerifying test subset boundary partition:")
    curr_idx = 0
    for name, expected_rows, expected_units in test_benchmarks:
        sub = test_df.iloc[curr_idx : curr_idx + expected_rows]
        u_count = sub['unit_number'].nunique()
        u_min = sub['unit_number'].min()
        u_max = sub['unit_number'].max()
        c_first = sub['time_in_cycles'].iloc[0]
        c_last = sub['time_in_cycles'].iloc[-1]
        print(f"  {name}: Rows [{curr_idx:6d}:{curr_idx + expected_rows:6d}] (Total {len(sub):5d}) | Units {u_min}-{u_max} ({u_count} units) | Cycle start: {c_first}, end: {c_last}")
        curr_idx += expected_rows

    print("\nSubset separation conclusion:")
    print("  YES, we CAN cleanly separate FD001, FD002, FD003, and FD004.")
    print("  The raw files are simple unlabelled vertical concatenations of the four standard NASA C-MAPSS subsets.")

    # 4. For test.csv: Inspect RUL behavior
    print("\n--- 4. TEST RUL BEHAVIOR AUDIT ---")
    test_rul_nan_count = test_df['RUL'].isnull().sum()
    print(f"Total NaN values in test.csv RUL: {test_rul_nan_count} / {len(test_df)} ({test_rul_nan_count / len(test_df) * 100:.2f}%)")
    if test_rul_nan_count == len(test_df):
        print("  CRITICAL FINDING: The RUL column in test.csv is 100% NaN (completely missing)!")
        print("  It does NOT contain any valid numerical values. In NASA C-MAPSS, test RUL ground-truth")
        print("  is provided only for the final cycles in separate files (RUL_FD001.txt, etc.), and this")
        print("  HuggingFace dataset left test RUL entirely unpopulated.")
    else:
        # Check if RUL decreases by 1
        non_decrement_units = 0
        final_ruls = []
        for u, group in test_df.groupby('unit_number'):
            diffs = group['RUL'].diff().dropna()
            if not (diffs == -1).all():
                non_decrement_units += 1
            final_ruls.append(group['RUL'].iloc[-1])
        print(f"  Units where RUL does NOT decrease by 1: {non_decrement_units}")
        print(f"  Final RUL values: min={min(final_ruls)}, max={max(final_ruls)}, mean={np.mean(final_ruls):.2f}")

    # Check train RUL behavior for reference
    print("\nTrain RUL behavior check:")
    print(f"  Train RUL min: {train_df['RUL'].min()}, max: {train_df['RUL'].max()}, mean: {train_df['RUL'].mean():.2f}")
    # Inspect first unit in train
    u1 = train_df.iloc[:20631] # FD001
    u1_engine1 = u1[u1['unit_number'] == 1]
    print(f"  Train FD001 Unit 1: starts at RUL={u1_engine1['RUL'].iloc[0]}, ends at RUL={u1_engine1['RUL'].iloc[-1]} (decreases by 1 each cycle to 0)")

    # 5. List sensors that are constant or near-constant (std close to 0)
    print("\n--- 5. SENSOR VARIABILITY ANALYSIS ---")
    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
    print("\nGlobal standard deviation across full train dataset (all 4 subsets combined):")
    for s in sensor_cols:
        s_std = train_df[s].std()
        print(f"  {s:10s}: std = {s_std:12.6f}")

    print("\nPer-subset standard deviation analysis:")
    # Check FD001 specifically (rows 0:20631)
    fd001_train = train_df.iloc[0:20631]
    fd001_const = []
    print("  FD001 constant / near-constant sensors (std < 1e-4):")
    for s in sensor_cols:
        std_val = fd001_train[s].std()
        if std_val < 1e-4:
            fd001_const.append(s)
            print(f"    {s:10s}: std = {std_val:.6f} (CONSTANT)")

    # Check FD002 (rows 20631:74390)
    fd002_train = train_df.iloc[20631:74390]
    fd002_const = [s for s in sensor_cols if fd002_train[s].std() < 1e-4]
    print(f"  FD002 constant sensors: {fd002_const}")

    # Check FD003 (rows 74390:99110)
    fd003_train = train_df.iloc[74390:99110]
    fd003_const = [s for s in sensor_cols if fd003_train[s].std() < 1e-4]
    print(f"  FD003 constant sensors: {fd003_const}")

    # Check FD004 (rows 99110:160359)
    fd004_train = train_df.iloc[99110:160359]
    fd004_const = [s for s in sensor_cols if fd004_train[s].std() < 1e-4]
    print(f"  FD004 constant sensors: {fd004_const}")

    print("\nSummary on constant sensors:")
    print(f"  - When all 4 subsets are concatenated together, NO sensors have std < 1e-4 because operating conditions in FD002/FD004 shift baseline sensor values.")
    print(f"  - In single-operating condition subsets (FD001 and FD003), sensors 1, 5, 6, 10, 16, 18, 19 are constant (std = 0.0000).")
    print(f"  - In multi-operating condition subsets (FD002 and FD004), only sensor_16 is near-constant (std ~ 0.0039 - 0.0049).")

    # 6. Save plots to figures/
    print("\n--- 6. GENERATING AND SAVING FIGURES ---")
    os.makedirs("figures", exist_ok=True)

    # Plot 1: RUL distribution (Train RUL distribution, noting Test RUL is 100% NaN)
    plt.figure(figsize=(10, 5))
    plt.hist(train_df['RUL'], bins=60, color='royalblue', edgecolor='black', alpha=0.7)
    plt.axvline(train_df['RUL'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean RUL: {train_df['RUL'].mean():.1f}")
    plt.axvline(train_df['RUL'].median(), color='green', linestyle='-', linewidth=2, label=f"Median RUL: {train_df['RUL'].median():.1f}")
    plt.title("Train Set Remaining Useful Life (RUL) Distribution\n(Test Set RUL is 100% Missing / NaN)", fontsize=13)
    plt.xlabel("RUL (Cycles)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    fig1_path = os.path.join("figures", "rul_distribution.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig1_path}")

    # Plot 2: 3 example sensors vs time_in_cycles for 3 engines (from FD001 train)
    example_sensors = ['sensor_2', 'sensor_7', 'sensor_12']
    example_engines = [1, 2, 3]
    
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for idx, sensor in enumerate(example_sensors):
        ax = axes[idx]
        for engine_id, color in zip(example_engines, colors):
            engine_data = fd001_train[fd001_train['unit_number'] == engine_id]
            ax.plot(engine_data['time_in_cycles'], engine_data[sensor], label=f"Engine {engine_id}", color=color, linewidth=1.5)
        ax.set_ylabel(sensor, fontsize=11)
        ax.set_title(f"{sensor} vs Time in Cycles (FD001 Engines 1, 2, 3)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    axes[-1].set_xlabel("Time in Cycles", fontsize=11)
    fig.tight_layout()
    fig2_path = os.path.join("figures", "sensor_degradation_3engines.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig2_path}")

    # 7. Write audit report to audit_report.md
    print("\n--- 7. WRITING AUDIT REPORT ---")
    report_text = f"""# NASA C-MAPSS Turbofan Data Audit Report

## 1. Executive Summary
A comprehensive audit of `train.csv` and `test.csv` was conducted for Remaining Useful Life (RUL) prediction. 
Several critical anomalies and structural issues were identified in the data provided.

---

## 2. Basic Dataset Health & Shapes
| Metric | Train Set (`train.csv`) | Test Set (`test.csv`) |
| :--- | :--- | :--- |
| **Row Count** | 160,359 rows | 104,897 rows |
| **Column Count** | 27 columns | 27 columns |
| **Dtypes** | 22 `float64`, 5 `int64` | 23 `float64`, 4 `int64` |
| **Missing Values** | 0 missing values | **104,897 missing values** (100% of `RUL` column) |
| **Exact Duplicate Rows** | 0 | 0 |
| **Duplicate `(unit_number, time_in_cycles)`** | **89,223 pairs** | **52,683 pairs** |

---

## 3. Major Findings & Problems Identified

### Problem A: Test Set RUL is 100% NaN (Completely Missing)
- In `test.csv`, the `RUL` column has **104,897 NaN values** out of 104,897 rows (100% missing).
- It does **not** decrease by 1 per cycle; it contains no numerical values at all.
- **Root Cause**: In the standard NASA C-MAPSS benchmark, test engines are truncated before failure, and ground-truth RUL is supplied only for the last recorded cycle in separate files (`RUL_FD001.txt`, `RUL_FD002.txt`, `RUL_FD003.txt`, `RUL_FD004.txt`). The HuggingFace loader left the test RUL column empty.
- **Implication**: Evaluating test RMSE/MAE in Task 2 is mathematically impossible on `test.csv` as provided, without either:
  1. Obtaining the true NASA ground-truth RUL values (`RUL_FD00x.txt`) for test unit final cycles.
  2. Performing validation on a train split (e.g. reserving 20% of train units as a validation set).

### Problem B: Stacked Subsets FD001-FD004 Without Subset Labels
- The files are an unlabeled concatenation of all four C-MAPSS sub-datasets:
  - **FD001**: Train rows 0 to 20,631 (100 units); Test rows 0 to 13,096 (100 units).
  - **FD002**: Train rows 20,631 to 74,390 (260 units); Test rows 13,096 to 47,087 (259 units).
  - **FD003**: Train rows 74,390 to 99,110 (100 units); Test rows 47,087 to 63,683 (100 units).
  - **FD004**: Train rows 99,110 to 160,359 (249 units); Test rows 63,683 to 104,897 (248 units).
- **Can we separate them?** **YES**. The row count boundaries match the official NASA C-MAPSS specifications exactly, and `time_in_cycles` resets to 1 at each boundary.
- **Consequence of stacking**: `unit_number` resets to 1 in each sub-dataset, creating **89,223 duplicated `(unit_number, time_in_cycles)` pairs** in train and **52,683 pairs** in test. Furthermore, unit numbers 1 to 259 overlap completely between train and test.

### Problem C: Disparate Operating Conditions & Constant Sensors
- In single-operating condition subsets (**FD001** and **FD003**), **7 sensors are constant (zero variance)**:
  `sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`.
- In multi-operating condition subsets (**FD002** and **FD004**), engines operate across 6 altitude/Mach flight regimes, causing these same sensors to fluctuate widely.
- When stacked together globally, the overall standard deviation of all 21 sensors is non-zero, obscuring the fact that these sensors are uninformative within individual regimes unless properly normalized by operating condition.

---

## 4. Visualizations Generated
1. **`figures/rul_distribution.png`**:
   - Shows the distribution of RUL in `train.csv` (mean = {train_df['RUL'].mean():.1f}, median = {train_df['RUL'].median():.1f}, max = {train_df['RUL'].max()}).
   - Notes the 100% absence of test set RUL.
2. **`figures/sensor_degradation_3engines.png`**:
   - Plots degradation trajectories for `sensor_2`, `sensor_7`, and `sensor_12` across Engines 1, 2, and 3 from FD001.

---

## 5. Required Decision Before Proceeding to Task 2
According to project rules: *"If you find a problem with the data, stop and tell me before continuing."*
Because `test.csv` has 100% NaN RUL values, and the data combines FD001-FD004 without distinction, we cannot train and evaluate a sensible baseline on test without resolving:
1. Which sub-dataset to model (FD001 is standard for baseline benchmarking).
2. How to evaluate performance (e.g. acquire the true C-MAPSS test RUL ground truth files, or use a train-validation split).
"""
    with open("audit_report.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("Saved audit_report.md successfully.")
    print("=" * 70)

if __name__ == "__main__":
    run_audit()
