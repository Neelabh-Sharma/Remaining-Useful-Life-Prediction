import os
import pandas as pd
import numpy as np

def prepare_data():
    raw_dir = "CMAPSSData"
    out_dir = "data_clean"
    os.makedirs(out_dir, exist_ok=True)

    subsets = ["FD001", "FD002", "FD003", "FD004"]
    expected_rows = {
        "FD001": (20631, 13096),
        "FD002": (53759, 33991),
        "FD003": (24720, 16596),
        "FD004": (61249, 41214)
    }

    cols = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    # Store verification records
    verification_records = []
    constant_sensors_exact = {}
    constant_sensors_numerical = {}
    distinct_op_conds = {}

    print("=" * 80)
    print("TASK 1: PREPARING NASA C-MAPSS DATA (FD001 - FD004)")
    print("=" * 80)

    for sub in subsets:
        train_file = os.path.join(raw_dir, f"train_{sub}.txt")
        test_file = os.path.join(raw_dir, f"test_{sub}.txt")
        rul_file = os.path.join(raw_dir, f"RUL_{sub}.txt")

        # Check file existence
        for f in [train_file, test_file, rul_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing required file: {f}")

        # 1. Load Train
        train_df = pd.read_csv(train_file, sep=r'\s+', header=None)
        train_df = train_df.dropna(how='all', axis=1)
        if train_df.shape[1] != 26:
            raise ValueError(f"{train_file} unexpected number of columns: {train_df.shape[1]}")
        train_df.columns = cols
        train_df['subset'] = sub

        # RUL and RUL_clipped calculation
        max_cycles = train_df.groupby('unit_number')['time_in_cycles'].transform('max')
        train_df['RUL'] = max_cycles - train_df['time_in_cycles']
        train_df['RUL_clipped'] = train_df['RUL'].clip(upper=125)

        # Save cleaned train
        train_out = os.path.join(out_dir, f"train_{sub}.csv")
        train_df.to_csv(train_out, index=False)

        # 2. Load Test
        test_df = pd.read_csv(test_file, sep=r'\s+', header=None)
        test_df = test_df.dropna(how='all', axis=1)
        if test_df.shape[1] != 26:
            raise ValueError(f"{test_file} unexpected number of columns: {test_df.shape[1]}")
        test_df.columns = cols
        test_df['subset'] = sub

        # Save cleaned test
        test_out = os.path.join(out_dir, f"test_{sub}.csv")
        test_df.to_csv(test_out, index=False)

        # 3. Load RUL
        rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None)
        rul_df = rul_df.dropna(how='all', axis=1)
        if rul_df.shape[1] != 1:
            raise ValueError(f"{rul_file} unexpected number of columns: {rul_df.shape[1]}")
        rul_values = [int(v) for v in rul_df.iloc[:, 0].tolist()]
        
        test_engines = test_df['unit_number'].unique()
        test_rul_df = pd.DataFrame({
            'unit_number': [int(u) for u in test_engines],
            'true_RUL': rul_values
        })
        test_rul_out = os.path.join(out_dir, f"test_RUL_{sub}.csv")
        test_rul_df.to_csv(test_rul_out, index=False)

        # 4. Verifications
        train_engines_cnt = train_df['unit_number'].nunique()
        test_engines_cnt = len(test_engines)
        train_rows = len(train_df)
        test_rows = len(test_df)
        rul_cnt = len(rul_values)

        exp_train_rows, exp_test_rows = expected_rows[sub]
        assert rul_cnt == test_engines_cnt, f"RUL count ({rul_cnt}) != test engine count ({test_engines_cnt}) in {sub}"
        assert not train_df.duplicated(subset=['unit_number', 'time_in_cycles']).any(), f"Duplicate (unit, cycle) in train {sub}"
        assert not test_df.duplicated(subset=['unit_number', 'time_in_cycles']).any(), f"Duplicate (unit, cycle) in test {sub}"
        assert train_rows == exp_train_rows, f"Train row mismatch in {sub}: {train_rows} vs expected {exp_train_rows}"
        assert test_rows == exp_test_rows, f"Test row mismatch in {sub}: {test_rows} vs expected {exp_test_rows}"

        verification_records.append({
            'subset': sub,
            'train_engines': train_engines_cnt,
            'test_engines': test_engines_cnt,
            'train_rows': train_rows,
            'test_rows': test_rows,
            'rul_values_count': rul_cnt,
            'expected_train_rows': exp_train_rows,
            'expected_test_rows': exp_test_rows,
            'status': 'PASS'
        })

        # 5. Constant sensors in train
        sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
        stds = train_df[sensor_cols].std()
        const_s_exact = stds[stds == 0].index.tolist()
        const_s_num = stds[stds < 1e-10].index.tolist()
        constant_sensors_exact[sub] = const_s_exact
        constant_sensors_numerical[sub] = const_s_num

        # Distinct operating conditions (op settings rounded to 1 decimal)
        op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        rounded_ops = train_df[op_cols].round(1).drop_duplicates()
        distinct_op_conds[sub] = len(rounded_ops)

    # Print Verification Table
    v_df = pd.DataFrame(verification_records)
    print("\n--- VERIFICATION TABLE ---")
    print(v_df.to_string(index=False))

    print("\n--- CONSTANT SENSORS (std=0 in Train) ---")
    for sub in ["FD001", "FD003"]:
        print(f"  {sub} (exact std == 0.0): {constant_sensors_exact[sub]} (Count: {len(constant_sensors_exact[sub])})")
        print(f"  {sub} (nunique == 1, accounting for float representation noise std < 1e-10): {constant_sensors_numerical[sub]} (Count: {len(constant_sensors_numerical[sub])})")

    print("\nConstant sensors in multi-condition subsets:")
    for sub in ["FD002", "FD004"]:
        print(f"  {sub} (std < 1e-10): {constant_sensors_numerical[sub]} (Count: {len(constant_sensors_numerical[sub])})")

    print("\n--- DISTINCT OPERATING CONDITIONS (rounded to 1 decimal) ---")
    for sub in subsets:
        print(f"  {sub}: {distinct_op_conds[sub]} distinct operating condition combinations")

    print("\nData preparation completed successfully.")
    print("=" * 80)

if __name__ == "__main__":
    prepare_data()
