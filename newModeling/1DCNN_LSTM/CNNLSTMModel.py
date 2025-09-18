import pickle

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from keras import layers, models, regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def create_cnn_lstm(input_shape):
    model = models.Sequential([
        layers.Conv1D(64, 5, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

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

benign = pd.read_csv("../../separate/Benign.csv")
syn = pd.read_csv("../../separate/Syn.csv")
udp = pd.read_csv("../../separate/UDP.csv")

df = pd.concat([benign, syn, udp], axis=0, ignore_index=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

features = ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
       'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'SYN Flag Count',
       'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Fwd PSH Flags',
       'Bwd Packets/s', 'Min Packet Length', 'Packet Length Mean',
       'RST Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
       'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
       'Subflow Fwd Bytes', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
       'Init_Win_bytes_backward']

label_mapping = {"BENIGN" : 0, "Syn": 1, "UDP": 1}
df["Label"] = df["Label"].map(label_mapping)

groups = df.groupby(pd.Grouper(key="Timestamp", freq="1min"))

window_size = 100

processed_data = []
for name, group in groups:
    result = pad(group, window_size)
    if isinstance(result, list):
        processed_data.extend(result)
    else:
        processed_data.append(result)

final_df = pd.concat(processed_data, ignore_index=True)

train = final_df.iloc[:131000]
test = final_df.iloc[131000:]

X_train = train[features].values
y_train = train["Label"].values
X_test = test[features].values
y_test = test["Label"].values

# X = df[features].values
# y = df["Label"].values

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.1,
#     shuffle=False
# )

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = RobustScaler(quantile_range=(5, 95))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open('imputer.pkl-pre-ReFeature', 'wb') as f:
    pickle.dump(imputer, f)
with open('scaler.pkl-pre-ReFeature', 'wb') as f:
    pickle.dump(scaler, f)

X_train, y_train = create_windows(X_train, y_train, window_size)
X_test, y_test = create_windows(X_test, y_test, window_size)

model = create_cnn_lstm((window_size, len(features)))

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3),
    ModelCheckpoint('1DCNN-LSTM-pre-ReFeature-Best.h5', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

model.save("1DCNN-LSTM-ReFeature-pre.h5")

y_pred_binary = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_binary))