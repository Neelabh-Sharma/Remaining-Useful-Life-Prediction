import pandas as pd
import numpy as np

df = pd.read_csv('results/e1_pilot_runs.csv')

print('--- PER-RUN TABLE ---')
for idx, r in df.iterrows():
    print(f"{r['condition']} | seed {int(r['seed'])} | Ep {int(r['epochs'])} | Val {r['best_val_rmse']:.2f} | TestA {r['rmse_A']:.2f} | TestB {r['rmse_B']:.2f} | NASA {r['nasa_score']:.1f} | {r['seconds']:.1f}s")

print('\n--- SUMMARY (MEAN +/- STD) ---')
summary_rows = []
for cond in ['C1', 'C2', 'C3', 'C4', 'C5']:
    sub = df[df['condition'] == cond]
    val_m, val_s = sub['best_val_rmse'].mean(), sub['best_val_rmse'].std()
    ta_m, ta_s = sub['rmse_A'].mean(), sub['rmse_A'].std()
    tb_m, tb_s = sub['rmse_B'].mean(), sub['rmse_B'].std()
    na_m, na_s = sub['nasa_score'].mean(), sub['nasa_score'].std()
    summary_rows.append({
        'condition': cond,
        'val_rmse_mean': val_m, 'val_rmse_std': val_s,
        'rmse_A_mean': ta_m, 'rmse_A_std': ta_s,
        'rmse_B_mean': tb_m, 'rmse_B_std': tb_s,
        'nasa_mean': na_m, 'nasa_std': na_s
    })
    print(f"{cond}: Val={val_m:.2f}+/-{val_s:.2f} | TestA={ta_m:.2f}+/-{ta_s:.2f} | TestB={tb_m:.2f}+/-{tb_s:.2f} | NASA={na_m:.1f}+/-{na_s:.1f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('results/e1_pilot_summary.csv', index=False)

print('\n--- PAIRED DIFFERENCES (PER SEED) ---')
c1_a = df[df['condition']=='C1'].sort_values('seed')['rmse_A'].values
c2_a = df[df['condition']=='C2'].sort_values('seed')['rmse_A'].values

for cond in ['C1', 'C2', 'C3', 'C4', 'C5']:
    c_a = df[df['condition']==cond].sort_values('seed')['rmse_A'].values
    diff_c1 = c_a - c1_a
    diff_c2 = c_a - c2_a
    print(f"{cond} raw TestA: {c_a.round(2).tolist()}")
    print(f"  vs C1 (Global): diff={[round(x, 2) for x in diff_c1]}, mean={diff_c1.mean():.2f}")
    print(f"  vs C2 (Cond):   diff={[round(x, 2) for x in diff_c2]}, mean={diff_c2.mean():.2f}")

# Check pooled std between C2 and C3/C4
s_c2 = df[df['condition']=='C2']['rmse_A'].std()
s_c3 = df[df['condition']=='C3']['rmse_A'].std()
s_c4 = df[df['condition']=='C4']['rmse_A'].std()

sp_c2_c3 = np.sqrt((2*s_c2**2 + 2*s_c3**2)/4)
sp_c2_c4 = np.sqrt((2*s_c2**2 + 2*s_c4**2)/4)

m_c2 = df[df['condition']=='C2']['rmse_A'].mean()
m_c3 = df[df['condition']=='C3']['rmse_A'].mean()
m_c4 = df[df['condition']=='C4']['rmse_A'].mean()

print(f"\nPooled std (C2, C3): {sp_c2_c3:.4f}, Mean diff (C3 - C2): {m_c3 - m_c2:.2f} -> {(m_c3 - m_c2)/sp_c2_c3:.2f} pooled std")
print(f"Pooled std (C2, C4): {sp_c2_c4:.4f}, Mean diff (C4 - C2): {m_c4 - m_c2:.2f} -> {(m_c4 - m_c2)/sp_c2_c4:.2f} pooled std")

print('\n--- RANKING DISAGREEMENTS (VAL vs TEST) ---')
for s in [0, 1, 2]:
    sub_s = df[df['seed']==s].copy()
    sub_s['rank_val'] = sub_s['best_val_rmse'].rank()
    sub_s['rank_test'] = sub_s['rmse_A'].rank()
    print(f"Seed {s}:")
    for _, r in sub_s.iterrows():
        disagree = "DISAGREE" if r['rank_val'] != r['rank_test'] else "AGREE"
        print(f"  {r['condition']}: Val={r['best_val_rmse']:.2f} (rank {int(r['rank_val'])}), TestA={r['rmse_A']:.2f} (rank {int(r['rank_test'])}) -> {disagree}")
