import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models

df = pd.read_csv("DDoS_dataset.csv")
df.columns = df.columns.str.strip()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")
features = ["Bwd IAT Min", "Total Length of Bwd Packets", "Fwd IAT Min", "Max Packet Length", "Average Packet Size",
            "Flow IAT Max", "Fwd Packets/s", "Bwd IAT Max", "Fwd Header Length", "Bwd IAT Std",
            "URG Flag Count", "Bwd Header Length", "Bwd IAT Mean", "Flow IAT Std", "Subflow Bwd Bytes",
            "Bwd Packet Length Max", "Bwd Packet Length Mean", "Avg Bwd Segment Size", "Total Fwd Packets", "Bwd IAT Total",
            "Subflow Fwd Packets", "Fwd IAT Mean", "Fwd Packet Length Std", "Init_Win_bytes_backward", "Subflow Fwd Bytes",
            "Subflow Bwd Packets", "Fwd IAT Total", "Fwd IAT Std", "act_data_pkt_fwd", "Avg Fwd Segment Size",
            "Total Length of Fwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Mean", "Init_Win_bytes_forward", "Fwd IAT Max", "Label"]
df = df[features]

labelMapping = {"BENIGN" : 0, "DDoS": 1}
df["Label"] = df["Label"].map(labelMapping)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

windowSize = 30
shape = (X.shape[0] - windowSize + 1, windowSize, X.shape[1])
strides = (X.strides[0], X.strides[0], X.strides[1])
X = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

y = y[windowSize-1:]

cutoff = int(len(df) * 0.9)
X_train = X[:cutoff]
y_train = y[:cutoff]

X_test = X[cutoff:]
y_test = y[cutoff:]

print("X_windows shape:", X.shape)  # (샘플수, 윈도우크기, 특징수)
print("X_train shape:", X_train.shape)

model = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', padding="same", input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu', padding="same"),
    layers.MaxPooling1D(2),
    layers.Conv1D(128, 3, activation='relu', padding="same"),
    layers.MaxPooling1D(2),
    layers.Conv1D(256, 3, activation='relu', padding="same"),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')