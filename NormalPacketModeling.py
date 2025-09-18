import pandas as pd
import numpy as np
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import pickle

def createWindows(data, windowSize):
    return np.array([data[i:i+windowSize] for i in range(len(data)-windowSize+1)])

def weighted_mse(y_true, y_pred):
    from tensorflow import constant
    weights = constant([1.0, 5.0, 5.0, 5.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0, 1.0, 1.0,
                         1.0, 5.0, 1.0, 1.0, 1.0])
    from tensorflow import reduce_mean
    return reduce_mean(weights * (y_true - y_pred)**2)

def create1DCNNAutoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)  # 30 → 15
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(3)(x)  # 15 → 5 (축소율 6배)

    # 디코더
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(3)(x)  # 5 → 15
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)  # 15 → 30

    outputs = layers.Conv1D(input_shape[-1], 3, activation='linear', padding='same')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(1e-3), loss=weighted_mse)
    return model

df = pd.read_csv("separate/Benign(M).csv")

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

features = ['Total Backward Packets', 'Total Length of Fwd Packets',
       'Fwd Packet Length Max', 'Fwd Packet Length Min',
       'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Min Packet Length',
       'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
       'Packet Length Variance', 'ACK Flag Count', 'URG Flag Count',
       'CWE Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size',
       'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       'Init_Win_bytes_backward', 'act_data_pkt_fwd']

X = df[features].replace([np.inf, -np.inf], np.nan).values

imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)
qt = QuantileTransformer(output_distribution='normal')
X = qt.fit_transform(X)

X_noisy = X + 0.05 * np.random.normal(size=X.shape)
X_combined = np.vstack([X, X_noisy])

scaler = RobustScaler(quantile_range=(5, 95))
X_scaled = scaler.fit_transform(X_combined)

with open('CNNAutoEncoder/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
with open('CNNAutoEncoder/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

windowSize = 30
X_window = createWindows(X_scaled, windowSize)

# 모델 학습
model = create1DCNNAutoencoder((windowSize, len(features)))
history = model.fit(
    X_window, X_window,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=15),
        ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)

# 개선된 임계값 계산
train_recon = model.predict(X_window)
sample_errors = np.mean((X_window - train_recon)**2, axis=(1,2))
threshold = np.mean(sample_errors) + 3*np.std(sample_errors)

mse = mean_squared_error(X_window.reshape(-1, X_window.shape[-1]), train_recon.reshape(-1, train_recon.shape[-1]))

feature_errors = np.mean((X_window - train_recon) ** 2, axis=(0, 1))

precision = np.mean(sample_errors > threshold)

print(mse)
print(feature_errors)
print(precision)

model.save("CNNAutoEncoder/model.h5")