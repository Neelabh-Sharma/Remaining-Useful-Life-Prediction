import numpy as np
import pandas as pd
from preprocess import get_preprocessed_data
from utils import make_train_windows, make_last_window, rmse, nasa_score
from sklearn.linear_model import Ridge

rows = []
for sub in ['FD002', 'FD004']:
    tr_raw = pd.read_csv(f'data_clean/train_{sub}.csv')
    te_raw = pd.read_csv(f'data_clean/test_{sub}.csv')
    te_rul = pd.read_csv(f'data_clean/test_RUL_{sub}.csv')['true_RUL'].values

    # Ridge Global
    tr_g, te_g, feats_g, _, _ = get_preprocessed_data(sub, mode='G')
    X_tr_g, y_tr_g = make_train_windows(tr_g, feats_g, window=30, stride=5)
    X_te_g, _ = make_last_window(te_g, feats_g, window=30)
    ridge_g = Ridge(alpha=100.0, random_state=0)
    ridge_g.fit(X_tr_g, y_tr_g)
    pred_g = np.clip(ridge_g.predict(X_te_g), 0, None)
    rmse_g_A = rmse(te_rul, pred_g)
    rmse_g_B = rmse(np.minimum(te_rul, 125.0), pred_g)
    nasa_g = nasa_score(te_rul, pred_g)

    # Ridge Cond
    tr_c, te_c, feats_c, _, _ = get_preprocessed_data(sub, mode='C')
    X_tr_c, y_tr_c = make_train_windows(tr_c, feats_c, window=30, stride=5)
    X_te_c, _ = make_last_window(te_c, feats_c, window=30)
    ridge_c = Ridge(alpha=100.0, random_state=0)
    ridge_c.fit(X_tr_c, y_tr_c)
    pred_c = np.clip(ridge_c.predict(X_te_c), 0, None)
    rmse_c_A = rmse(te_rul, pred_c)
    rmse_c_B = rmse(np.minimum(te_rul, 125.0), pred_c)
    nasa_c = nasa_score(te_rul, pred_c)

    rows.append({'subset': sub, 'model': 'Ridge', 'norm': 'Global', 'rmse_A': rmse_g_A, 'rmse_B': rmse_g_B, 'nasa': nasa_g})
    rows.append({'subset': sub, 'model': 'Ridge', 'norm': 'Cond', 'rmse_A': rmse_c_A, 'rmse_B': rmse_c_B, 'nasa': nasa_c})
    print(f'[{sub}] Ridge Global: RMSE(A)={rmse_g_A:.2f}, RMSE(B)={rmse_g_B:.2f}, NASA={nasa_g:.1f}')
    print(f'[{sub}] Ridge Cond:   RMSE(A)={rmse_c_A:.2f}, RMSE(B)={rmse_c_B:.2f}, NASA={nasa_c:.1f}')

df_ridge = pd.DataFrame(rows)
df_ridge.to_csv('results/tier1c_ridge_runs.csv', index=False)
print('Saved results/tier1c_ridge_runs.csv successfully!')
