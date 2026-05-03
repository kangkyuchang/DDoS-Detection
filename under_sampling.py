import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

def minibatch_kmeans_undersample(X, n_clusters):
    mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, random_state=42)
    mbk.fit(X)
    
    centers = mbk.cluster_centers_
    
    labels = mbk.labels_
    representative_indices = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            continue

        cluster_points = X[cluster_indices]
        distances = np.linalg.norm(cluster_points - centers[i], axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        representative_indices.append(closest_index)

    expanded_indices = set()
    count = 0
    for center_idx in representative_indices:
        count += 1
        if(len(expanded_indices) + (len(representative_indices) - count) >= 10000):
            window_size = 0
        else:
            window_size = 1
        start = max(center_idx - window_size, 0)
        end = min(center_idx + window_size + 1, len(X))
        expanded_indices.update(range(start, end))

    expanded_indices = sorted(expanded_indices)
    return expanded_indices

fileName = ["LDAP", "MSSQL", "NetBIOS", "Portmap", "Syn", "UDP", "UDPLag"]
for name in fileName:
    filePath = f"dataset/03-11/{name}.csv"
    data = pd.read_csv(filePath)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    data = data.sort_values(by=" Timestamp")

    data = data[data[" Label"] == name]

    X = data.iloc[:, 8:-3].values
    y = data.iloc[:, -1].values

    us = minibatch_kmeans_undersample(X, n_clusters=10000)
    us = np.unique(us)
    print(us.size)
    selected_data = data.iloc[us]
    selected_data.to_csv(f"./reundersample/undersampled_{name}.csv", index=False)