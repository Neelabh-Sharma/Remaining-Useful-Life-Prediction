# NASA C-MAPSS Turbofan Data Audit Report

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
   - Shows the distribution of RUL in `train.csv` (mean = 122.3, median = 113.0, max = 542).
   - Notes the 100% absence of test set RUL.
2. **`figures/sensor_degradation_3engines.png`**:
   - Plots degradation trajectories for `sensor_2`, `sensor_7`, and `sensor_12` across Engines 1, 2, and 3 from FD001.

---

## 5. Required Decision Before Proceeding to Task 2
According to project rules: *"If you find a problem with the data, stop and tell me before continuing."*
Because `test.csv` has 100% NaN RUL values, and the data combines FD001-FD004 without distinction, we cannot train and evaluate a sensible baseline on test without resolving:
1. Which sub-dataset to model (FD001 is standard for baseline benchmarking).
2. How to evaluate performance (e.g. acquire the true C-MAPSS test RUL ground truth files, or use a train-validation split).
