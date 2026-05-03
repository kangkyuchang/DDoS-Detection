import keras
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from keras import layers, models, regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import ZeroPadding1D

import winsplit as ws

def create1DCNNAutoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(5)(x)

    x = layers.Conv1D(128, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(5)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)

    outputs = layers.Conv1D(input_shape[-1], 3, activation='linear', padding='same')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-4), loss="mse")
    return model

df = pd.read_csv("../../dataset/BENIGN.csv")
drop_column = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', "SimillarHTTP", 'Inbound']
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)
df.replace(np.inf, np.nan, inplace=True)
# df["Label"] = df["Label"].map(lambda x: 0 if x == "BENIGN" else 1)
# df.drop(columns=drop_column, inplace=True)

train, test = ws.spilt_by_label(df)
train["Timestamp"] = pd.to_datetime(train["Timestamp"])
train = train.sort_values("Timestamp").reset_index(drop=True)
test["Timestamp"] = pd.to_datetime(test["Timestamp"])
test = test.sort_values("Timestamp").reset_index(drop=True)
train.drop(columns=drop_column, inplace=True)
test.drop(columns=drop_column, inplace=True)
train["Label"] = train["Label"].map(lambda x: 0 if x == "BENIGN" else 1)
test["Label"] = test["Label"].map(lambda x: 0 if x == "BENIGN" else 1)

# train_max = df.max()

train_max = train.max()

# train = df.fillna(train_max)

train = train.fillna(train_max)
test = test.fillna(train_max)

train_X = train.iloc[:, :-1].values
train_y = train.iloc[:, -1].values
test_X = test.iloc[:, :-1].values
test_y = test.iloc[:, -1].values

scaler = RobustScaler(quantile_range=(5, 95))
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

scaler_minmax = MinMaxScaler()
train_X = scaler_minmax.fit_transform(train_X)
test_X = scaler_minmax.transform(test_X)

# with open("max_values_autoencoder.pkl", "wb") as f:
#     pickle.dump(train_max, f)
# with open("scaler_autoencoder.pkl", "wb") as f:
#     pickle.dump(scaler, f)
# with open("scaler_minmax_autoencoder.pkl", "wb") as f:
#     pickle.dump(scaler, f)

window_size = 100
train_windows_X, train_windows_y = ws.create_windows(train_X, train_y, window_size, 40)
test_windows_X, test_windows_y = ws.create_windows(test_X, test_y, window_size, 40)

print(train_windows_X.shape, train_windows_y.shape)

model = create1DCNNAutoencoder(train_windows_X.shape[1:])

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min',restore_best_weights=True, min_delta=0.0001)

history = model.fit(
    train_windows_X, train_windows_X,
    epochs=78, #78
    batch_size=32
    # validation_data=(test_windows_X, test_windows_X),
    # callbacks=[early_stop]
)

y_pred_proba = model.predict(test_windows_X)
errors = np.mean(np.square(test_windows_X - y_pred_proba), axis=(1,2))
threshold = np.percentile(errors, 97)

print(f"설정된 임계치: {threshold}")

# model.save("DDoS_detection_autoencoder.h5")

