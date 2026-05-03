import keras
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from keras import layers, models, regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

import winsplit as ws

df = pd.read_csv("../../dataset/DDoSDataset.csv")
drop_column = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', "SimillarHTTP", 'Inbound']
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)
df.replace(np.inf, np.nan, inplace=True)
df["Label"] = df["Label"].map(lambda x: 0 if x == "BENIGN" else 1)
df.drop(columns=drop_column, inplace=True)

# train, test = ws.spilt_by_label(df)
# train["Timestamp"] = pd.to_datetime(train["Timestamp"])
# train = train.sort_values("Timestamp").reset_index(drop=True)
# test["Timestamp"] = pd.to_datetime(test["Timestamp"])
# test = test.sort_values("Timestamp").reset_index(drop=True)
# train.drop(columns=drop_column, inplace=True)
# test.drop(columns=drop_column, inplace=True)
# train["Label"] = train["Label"].map(lambda x: 0 if x == "BENIGN" else 1)
# test["Label"] = test["Label"].map(lambda x: 0 if x == "BENIGN" else 1)

train_max = df.max()

# train_max = train.max()

train = df.fillna(train_max)

# train = train.fillna(train_max)
# test = test.fillna(train_max)

train_X = train.iloc[:, :-1].values
train_y = train.iloc[:, -1].values
# test_X = test.iloc[:, :-1].values
# test_y = test.iloc[:, -1].values

scaler = RobustScaler(quantile_range=(5, 95))
train_X = scaler.fit_transform(train_X)
# test_X = scaler.transform(test_X)

with open("max_values_lstm.pkl", "wb") as f:
    pickle.dump(train_max, f)
with open("scaler_lstm.pkl", "wb") as f:
    pickle.dump(scaler, f)

window_size = 100
train_windows_X, train_windows_y = ws.create_windows(train_X, train_y, window_size, 40)
# test_windows_X, test_windows_y = ws.create_windows(test_X, test_y, window_size, 40)

train_windows_y = train_windows_y.reshape(-1, 1)
# test_windows_y = test_windows_y.reshape(-1, 1)

print(train_windows_X.shape, train_windows_y.shape)

model = models.Sequential([
    layers.Conv1D(64, 5, activation='relu', padding='same', input_shape=train_windows_X.shape[1:]),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.2),

    layers.LSTM(128, return_sequences=True),
    layers.LSTM(64),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min',restore_best_weights=True, min_delta=0.0001)

metrics = [
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.F1Score(name='f1_score', dtype=None, threshold=0.5)
]

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=metrics
)

history = model.fit(
    train_windows_X, train_windows_y,
    epochs=2, #2
    batch_size=32,
    # validation_data=(test_windows_X, test_windows_y),
    # callbacks=[early_stop]
)

model.save("DDoS_detection_lstm.h5")

