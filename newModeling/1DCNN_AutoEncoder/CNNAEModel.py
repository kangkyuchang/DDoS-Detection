import pickle
import numpy as np
import pandas as pd
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.src.layers import ZeroPadding1D

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from keras import layers, models
from keras.optimizers import Adam
import matplotlib.pyplot as plt


def create_windows(data, labels, window_size):
    num_samples = data.shape[0] - window_size + 1
    X = np.array([data[i:i+window_size] for i in range(num_samples)])
    y = np.array([1 if 1 in labels[i:i+window_size] else 0 for i in range(num_samples)])
    return X, y

def create1DCNNAutoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(3)(x)

    # 디코더
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(3)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    x = ZeroPadding1D(2)(x)

    outputs = layers.Conv1D(input_shape[-1], 3, activation='linear', padding='same')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

df = pd.read_csv("../../separate/Benign.csv")
# syn = pd.read_csv("../../separate/Syn.csv")
# udp = pd.read_csv("../../separate/UDP.csv")

# df = pd.concat([benign, syn, udp], axis=0, ignore_index=True)
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

label_mapping = {"BENIGN" : 0}
df["Label"] = df["Label"].map(label_mapping)

X = df[features].values
y = df["Label"].values

# X_train, X_test, y_train, y_test = train_test_split (
#     X, y,
#     test_size=0.1,
#     shuffle=False
# )

imputer = SimpleImputer(strategy="median")
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

model = create1DCNNAutoencoder((window_size, len(features)))
history = model.fit(
    X_train, X_train,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint('1DCNN-AutoEncoder-ReFeature-Best.h5', save_best_only=True)
    ]
)

train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
print(f"Training Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

# X_test_pred = model.predict(X_test)
# reconstruct_errors = np.mean(np.square(X_test - X_test_pred), axis=(1,2))

train_pred = model.predict(X_train)
train_errors = np.mean(np.square(X_train - train_pred), axis=(1,2))
train_errors = train_errors[train_errors < 1]
threshold = np.mean(train_errors) + 2 * np.std(train_errors)

print(f"size: {train_errors.size}")

#
# np.save("threshold.npy", threshold)
print(f"mean: {np.mean(train_errors):.4f}")
print(f"std: {np.std(train_errors):.4f}")
print(f"threshold: {threshold:.4f}")

# y_pred = (reconstruct_errors > threshold).astype(int)

# precision = precision_score(y_test, y_pred)
#
mse = np.mean(np.square(X_train - train_pred))
print(f"mse: {mse:.4f}")
# print(f"Precision: {precision:.4f}")

# step = 0.01
# start = np.floor(train_errors.min() / step) * step
# stop = np.floor(train_errors.max() / step)
bin_edges = np.arange(0, 1, 0.01)

# np.savetxt("../../output.txt", train_errors, fmt="%.4f", delimiter=",")

plt.figure(figsize=(12, 4))
plt.hist(train_errors, bins=bin_edges, color="skyblue", edgecolor="black")
plt.xticks(bin_edges)
plt.axvline(threshold, color="red", linestyle="dashed", label=f"Threshold: {threshold:.4f}")
plt.legend()
plt.xlabel("Reconstruction Error (MSE)")
plt.ylabel("Number of samples")
plt.title("Normal Data Reconstruction Error")
plt.show()
model.save("1DCNN-AutoEncoder-ReFeature.h5")
