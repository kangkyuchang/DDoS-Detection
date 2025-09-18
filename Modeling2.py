import pandas as pd
import numpy as np
from keras import layers, models, regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pickle


features = ['Total Backward Packets', 'Total Length of Fwd Packets',
       'SYN Flag Count', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
       'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Min Packet Length',
       'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
       'Packet Length Variance', 'ACK Flag Count', 'URG Flag Count',
       'CWE Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size',
       'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Init_Win_bytes_forward',
       'Init_Win_bytes_backward', 'act_data_pkt_fwd']

benign = pd.read_csv("separate/Benign(M).csv")
syn = pd.read_csv("separate/Syn.csv")
udp = pd.read_csv("separate/UDP.csv")

df = pd.concat([benign, syn, udp], axis=0, ignore_index=True)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

label_mapping = {"BENIGN": 0, "Syn": 1, "UDP": 2}
df["Label"] = df["Label"].map(label_mapping)

split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(train_df[features].replace([np.inf, -np.inf], np.nan))

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

X_test_imputed = imputer.transform(test_df[features].replace([np.inf, -np.inf], np.nan))
X_test_scaled = scaler.transform(X_test_imputed)

with open('CNNModelF21/imputer-CL.pkl', 'wb') as f:
    pickle.dump(imputer, f)
with open('CNNModelF21/scaler-CL.pkl', 'wb') as f:
    pickle.dump(scaler, f)


def create_windows(data, labels, window_size):
    num_samples = data.shape[0] - window_size + 1
    X = np.array([data[i:i+window_size] for i in range(num_samples)])
    y = np.array([1 if 1 in labels[i:i+window_size] else 0 if 2 in labels[i:i+window_size] else 0
                 for i in range(num_samples)])
    return X, y


window_size = 30

X_train, y_train = create_windows(X_train_scaled, train_df["Label"].values, window_size)

X_test, y_test = create_windows(X_test_scaled, test_df["Label"].values, window_size)

def create_cnn_lstm(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(64, 5, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_cnn_lstm((window_size, len(features)), 3)

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
)

from sklearn.metrics import classification_report

y_pred = model.predict(X_test).argmax(axis=1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.4f}')

print(classification_report(y_test, y_pred))

model.save('CNNModelF21/CNN-LSTMModel.h5')
