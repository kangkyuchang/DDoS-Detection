import pandas as pd
import numpy as np
from keras import layers, models, regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import pickle


def createWindows(data, windowSize):
    shape = (data.shape[0] - windowSize + 1, windowSize, data.shape[1])
    strides = (data.strides[0], data.strides[0], data.strides[1])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

# df = pd.read_csv("DDoS_dataset.csv")
# df.columns = df.columns.str.strip()
# df["Timestamp"] = pd.to_datetime(df["Timestamp"])
# df = df.sort_values("Timestamp")
# features = ["Bwd IAT Min", "Total Length of Bwd Packets", "Fwd IAT Min", "Max Packet Length", "Average Packet Size",
#             "Flow IAT Max", "Fwd Packets/s", "Bwd IAT Max", "Fwd Header Length", "Bwd IAT Std",
#             "URG Flag Count", "Bwd Header Length", "Bwd IAT Mean", "Flow IAT Std", "Subflow Bwd Bytes",
#             "Bwd Packet Length Max", "Bwd Packet Length Mean", "Avg Bwd Segment Size", "Total Fwd Packets", "Bwd IAT Total",
#             "Subflow Fwd Packets", "Fwd IAT Mean", "Fwd Packet Length Std", "Init_Win_bytes_backward", "Subflow Fwd Bytes",
#             "Subflow Bwd Packets", "Fwd IAT Total", "Fwd IAT Std", "act_data_pkt_fwd", "Avg Fwd Segment Size",
#             "Total Length of Fwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Mean", "Init_Win_bytes_forward", "Fwd IAT Max", "Label"]
# df = df[features]
#
# labelMapping = {"BENIGN" : 0, "DDoS": 1}
# df["Label"] = df["Label"].map(labelMapping)
#
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

benign = pd.read_csv("separate/Benign(M).csv")
syn = pd.read_csv("separate/Syn.csv")
udp = pd.read_csv("separate/UDP.csv")

df = pd.concat([benign, syn, udp], axis=0, ignore_index=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")

# features = ['Fwd Packet Length Mean', 'Init_Win_bytes_forward', 'Min Packet Length', 'Fwd Packet Length Min', 'Fwd Packet Length Max',
#        'Avg Fwd Segment Size', 'ACK Flag Count', 'Max Packet Length', 'Average Packet Size', 'Subflow Fwd Bytes', 'Packet Length Mean',
#        'Packet Length Variance', 'Total Length of Fwd Packets','Packet Length Std', 'URG Flag Count', 'act_data_pkt_fwd',
#        'Bwd Header Length', 'Avg Bwd Segment Size', 'Total Backward Packets', 'Init_Win_bytes_backward', 'Fwd Packet Length Std', 'CWE Flag Count',
#        'min_seg_size_forward', 'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Bwd Packet Length Min', 'Bwd Packets/s',
#        'Subflow Bwd Bytes', 'Total Length of Bwd Packets', 'Flow Bytes/s', 'Bwd IAT Max', 'Bwd IAT Total', 'Bwd IAT Min', 'Bwd IAT Mean',
#        'Fwd Header Length.1', 'Bwd IAT Std', 'Total Fwd Packets', 'Fwd PSH Flags', 'Flow Packets/s', 'Subflow Bwd Packets',
#        'Flow IAT Min', 'Flow IAT Max', 'Subflow Fwd Packets', 'Down/Up Ratio', 'Flow IAT Mean', 'Fwd Header Length', 'Fwd IAT Min', 'Fwd IAT Total',
#        'Fwd IAT Mean', 'RST Flag Count', 'Fwd Packets/s', 'Fwd IAT Max', 'Fwd IAT Std', 'Flow IAT Std', 'Idle Std', 'Active Min',
#        'SYN Flag Count', 'Active Mean', 'Idle Max', 'Active Max', 'Active Std', 'Idle Min', 'Idle Mean', 'Bwd Packet Length Std', "Label"]

features = ['Total Backward Packets', 'Total Length of Fwd Packets',
       'Fwd Packet Length Max', 'Fwd Packet Length Min',
       'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Min Packet Length',
       'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
       'Packet Length Variance', 'ACK Flag Count', 'URG Flag Count',
       'CWE Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size',
       'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'Label']

df = df[features]

labelMapping = {"BENIGN" : 0, "Syn": 1, "UDP": 2}
df["Label"] = df["Label"].map(labelMapping)

df = df.replace([np.inf, -np.inf], np.nan)

X = df.iloc[:, : -1].values
y = df.iloc[:, -1].values

cutoff = int(len(df) * 0.8)
X_train = X[:cutoff]
y_train = y[:cutoff]

X_test = X[cutoff:]
y_test = y[cutoff:]

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
# X_train[["Flow Bytes/s", "Flow Packets/s"]] = imputer.fit_transform(X_train[["Flow Bytes/s", "Flow Packets/s"]])
# X_test[["Flow Bytes/s", "Flow Packets/s"]] = imputer.transform(X_test[["Flow Bytes/s", "Flow Packets/s"]])

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open('CNNModelF21/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
with open('CNNModelF21/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

windowSize = 30
X_train_window = createWindows(X_train, windowSize)
X_test_window = createWindows(X_test, windowSize)

y_train = y_train[windowSize-1:]
y_test = y_test[windowSize-1:]

model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', padding="same", input_shape=(X_train_window.shape[1], X_train_window.shape[2])),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),

    layers.Conv1D(128, 3, activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(len(np.unique(y_train)), activation='softmax')
])

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train_window, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_window, y_test),
    callbacks=[early_stop, lr_scheduler]
)

loss, accuracy = model.evaluate(X_test_window, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

model.save("CNNModelF21/CNNModelF21-Class2.h5")