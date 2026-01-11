import pandas as pd
import numpy as np

def spilt_by_label(data):
    labels = data.iloc[:, -1].unique()
    train = pd.DataFrame()
    test = pd.DataFrame()
    for label in labels:
        df = data[data["Label"] == label]
        split_size = int(len(df) * 0.8)
        if train.empty & test.empty:
            train, test = df.iloc[:split_size], df.iloc[split_size:]
        else:
            train = pd.concat([train, df.iloc[:split_size]])
            test = pd.concat([test, df.iloc[split_size:]])
    return train, test

def create_windows(data, labels, window_size, labeling_ratio=0):
    threshold = 1 if labeling_ratio == 0 else int(window_size * (labeling_ratio / 100))
    print('threshold', threshold)
    num_samples = data.shape[0] - window_size + 1
    X = np.array([data[i:i+window_size] for i in range(num_samples)])
    y = np.array([
        1 if np.sum(labels[i:i + window_size] == 1) >= threshold else 0
        for i in range(num_samples)
    ])
    return X, y
