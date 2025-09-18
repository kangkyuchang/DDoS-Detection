from keras import models
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
import pickle
from keras.optimizers import Adam

def pad(data, window_size):
    if len(data) < window_size:
        padding_rows = window_size - len(data)
        padding = pd.DataFrame(np.zeros((padding_rows, data.shape[1]), dtype=int), columns=data.columns)
        return pd.concat([data, padding], ignore_index=True)
    else :
        chunk = [data[i:i+window_size].reset_index(drop=True) for i in range(0, len(data), window_size)]
        if len(chunk[-1]) < window_size:
            padding_rows = window_size - len(chunk[-1])
            padding = pd.DataFrame(np.zeros((padding_rows, data.shape[1]), dtype=int), columns=data.columns)
            chunk[-1] = pd.concat([chunk[-1], padding], ignore_index=True)
        return chunk

def create_windows(data, labels, window_size):
    num_samples = data.shape[0] - window_size + 1
    X = np.array([data[i:i+window_size] for i in range(num_samples)])
    y = np.array([1 if 1 in labels[i:i+window_size] else 0 for i in range(num_samples)])
    return X, y

path = "newModeling/1DCNN_LSTM/"

model = models.load_model(path + "1DCNN-LSTM-ReFeature-pre.h5")
# model.compile(optimizer=Adam(1e-3), loss="mse")

df = pd.read_csv("DDoS_dataset.csv")
df.columns = df.columns.str.strip()
# features = ['Fwd Packet Length Mean', 'Init_Win_bytes_forward', 'Min Packet Length', 'Fwd Packet Length Min', 'Fwd Packet Length Max',
#        'Avg Fwd Segment Size', 'ACK Flag Count', 'Max Packet Length', 'Average Packet Size', 'Subflow Fwd Bytes', 'Packet Length Mean',
#        'Packet Length Variance', 'Total Length of Fwd Packets','Packet Length Std', 'URG Flag Count', 'act_data_pkt_fwd',
#        'Bwd Header Length', 'Avg Bwd Segment Size', 'Total Backward Packets', 'Init_Win_bytes_backward', 'Fwd Packet Length Std', 'CWE Flag Count',
#        'min_seg_size_forward', 'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Bwd Packet Length Min', 'Bwd Packets/s',
#        'Subflow Bwd Bytes', 'Total Length of Bwd Packets', 'Flow Bytes/s', 'Bwd IAT Max', 'Bwd IAT Total', 'Bwd IAT Min', 'Bwd IAT Mean',
#        'Fwd Header Length.1', 'Bwd IAT Std', 'Total Fwd Packets', 'Fwd PSH Flags', 'Flow Packets/s', 'Subflow Bwd Packets',
#        'Flow IAT Min', 'Flow IAT Max', 'Subflow Fwd Packets', 'Down/Up Ratio', 'Flow IAT Mean', 'Fwd Header Length', 'Fwd IAT Min', 'Fwd IAT Total',
#        'Fwd IAT Mean', 'RST Flag Count', 'Fwd Packets/s', 'Fwd IAT Max', 'Fwd IAT Std', 'Flow IAT Std', 'Idle Std', 'Active Min',
#        'SYN Flag Count', 'Active Mean', 'Idle Max', 'Active Max', 'Active Std', 'Idle Min', 'Idle Mean', 'Bwd Packet Length Std']

# features = ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
#        'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'SYN Flag Count',
#        'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Fwd PSH Flags',
#        'Min Packet Length', 'RST Flag Count', 'ACK Flag Count',
#        'URG Flag Count', 'CWE Flag Count', 'Average Packet Size',
#        'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
#        'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd']

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

features = ['Total Backward Packets', 'Total Length of Fwd Packets',
       'SYN Flag Count', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
       'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Min Packet Length',
       'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
       'Packet Length Variance', 'ACK Flag Count', 'URG Flag Count',
       'CWE Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size',
       'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       'Init_Win_bytes_backward', 'act_data_pkt_fwd']

df = df.replace([np.inf, -np.inf], np.nan)

# df = df[df["Label"] == "BENIGN"]

labelMapping = {"BENIGN" : 0, "Syn": 1, "UDP": 1, "UDPLag": 1, "DDoS": 1}
df["Label"] = df["Label"].map(labelMapping)

windowSize = 100

# groups = df.groupby(pd.Grouper(key="Timestamp", freq="1min"))
#
# processed_data = []
# for name, group in groups:
#     result = pad(group, windowSize)
#     if isinstance(result, list):
#         processed_data.extend(result)
#     else:
#         processed_data.append(result)
#
# final_df = pd.concat(processed_data, ignore_index=True)
#
# X = final_df[features].values
# y = final_df["Label"].values

X = df[features].values
y = df["Label"].values

# X = X[100000:300000]
# y = y[100000:300000]

with open(path + 'imputer.pkl-pre-ReFeature', 'rb') as f:
    imputer = pickle.load(f)
with open(path + 'scaler.pkl-pre-ReFeature', 'rb') as f:
    scaler = pickle.load(f)

X = imputer.transform(X)
X = scaler.transform(X)

X_test, y_test = create_windows(X, y, windowSize)

print(y_test)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

# X_test_pred = model.predict(X_test)
# reconstruct_errors = np.mean(np.square(X_test - X_test_pred), axis=(1,2))

# np.savetxt("output.txt", reconstruct_errors, delimiter=",")

# threshold = 0.2
#
# print(threshold)
#
# y_pred = (reconstruct_errors > threshold).astype(int)
#
# precision = precision_score(y_test, y_pred)

# print(f"Precision: {precision:.4f}")