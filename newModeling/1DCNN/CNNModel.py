import pickle

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from keras import layers, models, regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def create_windows(data, labels, window_size):
    num_samples = data.shape[0] - window_size + 1
    X = np.array([data[i:i+window_size] for i in range(num_samples)])
    y = np.array([1 if 1 in labels[i:i+window_size] else 0 for i in range(num_samples)])
    return X, y

benign = pd.read_csv("../../separate/Benign.csv")
syn = pd.read_csv("../../separate/Syn.csv")
udp = pd.read_csv("../../separate/UDP.csv")

df = pd.concat([benign, syn, udp], axis=0, ignore_index=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

features = ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
       'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'SYN Flag Count',
       'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Fwd PSH Flags',
       'Min Packet Length', 'RST Flag Count', 'ACK Flag Count',
       'URG Flag Count', 'CWE Flag Count', 'Average Packet Size',
       'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
       'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd']

label_mapping = {"BENIGN" : 0, "Syn": 1, "UDP": 1}
df["Label"] = df["Label"].map(label_mapping)

X = df[features].values
y = df["Label"].values

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.1,
#     shuffle=False
# )

imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X)
# X_test = imputer.transform(X_test)

scaler = RobustScaler(quantile_range=(5, 95))
X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

with open('imputer.pkl-ReFeature', 'wb') as f:
    pickle.dump(imputer, f)
with open('scaler.pkl-ReFeature', 'wb') as f:
    pickle.dump(scaler, f)

window_size = 100
X_train, y_train = create_windows(X_train, y, window_size)
# X_test, y_test = create_windows(X_test, y_test, window_size)

model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', padding="same", input_shape=(X_train.shape[1], X_train.shape[2])),
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
    layers.Dense(1, activation='sigmoid')
])

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    # validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_scheduler, ModelCheckpoint('1D-CNN-ReFeature.h5', save_best_only=True)]
)

# loss, accuracy = model.evaluate(X_test, y_test)
#
# print(f'Test Accuracy: {accuracy:.4f}')
# print(f'Test Loss: {loss:.4f}')

# model.save("1D-CNN-pre.h5")

# print(classification_report(y_test, model.predict(X_test)))