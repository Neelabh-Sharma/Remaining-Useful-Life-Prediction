import pandas as pd
from sklearn.cluster import KMeans

train_df = pd.read_csv("data_clean/train_FD004.csv")
op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']

kmeans = KMeans(n_clusters=6, random_state=0, n_init=10)
kmeans.fit(train_df[op_cols])
train_df['cluster'] = kmeans.predict(train_df[op_cols])

print("=" * 60)
print("FD004 KMeans (k=6) Clusters on Operating Settings")
print("=" * 60)
counts = train_df['cluster'].value_counts().sort_index()
print(f"{'Cluster':<10} | {'Count':<10} | {'Percentage':<12} | {'Center (op_1, op_2, op_3)'}")
print("-" * 60)
for c in range(6):
    cnt = counts[c]
    pct = (cnt / len(train_df)) * 100
    center = kmeans.cluster_centers_[c]
    print(f"Cluster {c:<2} | {cnt:<10} | {pct:<11.2f}% | [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.1f}]")
print("=" * 60)

