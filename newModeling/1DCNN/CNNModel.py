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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report


def create_windows(data, labels, window_size, labeling_ratio=40):
    threshold = 1 if labeling_ratio == 0 else int(window_size * (labeling_ratio / 100))
    print('threshold', threshold)
    num_samples = data.shape[0] - window_size + 1
    X = np.array([data[i:i+window_size] for i in range(num_samples)])
    y = np.array([
        1 if np.sum(labels[i:i + window_size] == 1) >= threshold else 0
        for i in range(num_samples)
    ])
    return X, y

df = pd.read_csv("../../DDoSdataset.csv")
df.columns = df.columns.str.strip()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)
df["Label"] = df["Label"].map(lambda x: 0 if x == "BENIGN" else 1)

evaluate_df = pd.read_csv("../../dataset/DDoS_evaluation.csv")
evaluate_df.columns = evaluate_df.columns.str.strip()
evaluate_df["Timestamp"] = pd.to_datetime(evaluate_df["Timestamp"])
evaluate_df = evaluate_df.sort_values("Timestamp").reset_index(drop=True)
evaluate_df = evaluate_df.replace([np.inf, -np.inf], np.nan)
evaluate_df["Label"] = evaluate_df["Label"].map(lambda x: 0 if x == "BENIGN" else 1)

features = ['Init_Win_bytes_forward', 'Flow Packets/s', 'Bwd Packet Length Mean', 'Packet Length Variance',
            'Fwd Header Length', 'Max Packet Length', 'Fwd Packet Length Min', 'Packet Length Mean',
            'Subflow Fwd Bytes', 'Bwd IAT Min', 'Bwd Header Length', 'Average Packet Size',
            'Flow IAT Std', 'Fwd Packets/s', 'Total Length of Bwd Packets', 'Total Backward Packets', 'ACK Flag Count',
            'Fwd IAT Max', 'Bwd Packet Length Max', 'Fwd Packet Length Mean']

X_train = df[features].values
y_train = df["Label"].values
X_eval = evaluate_df[features].values
y_eval = evaluate_df["Label"].values

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_eval = imputer.transform(X_eval)

scaler = RobustScaler(quantile_range=(5, 95))
X_train = scaler.fit_transform(X_train)
X_eval = scaler.transform(X_eval)

with open('../../2025_10_20/WS_5/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
with open('../../2025_10_20/WS_5/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

window_size = 5
X_train, y_train = create_windows(X_train, y_train, window_size)
X_eval, y_eval = create_windows(X_eval, y_eval, window_size)

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
    callbacks=[early_stop, lr_scheduler, ModelCheckpoint('../../2025_10_20/WS_5/1D_CNN.h5', save_best_only=True)]
)

y_pred_proba = model.predict(X_eval)
y_pred = (y_pred_proba > 0.5).astype(int)

print("### 분류 평가 지표 ###")

# 1. 정확도 (Accuracy)
accuracy = accuracy_score(y_eval, y_pred)
print(f"정확도 (Accuracy): {accuracy:.4f}")

precision = precision_score(y_eval, y_pred, average='binary')
recall = recall_score(y_eval, y_pred, average='binary')
f1 = f1_score(y_eval, y_pred, average='binary')
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}")
print(f"F1-점수 (F1-Score): {f1:.4f}")

roc_auc = roc_auc_score(y_eval, y_pred_proba)
print(f"ROC AUC: {roc_auc:.4f}")

conf_matrix = confusion_matrix(y_eval, y_pred)
print("\n혼동 행렬 (Confusion Matrix):")
print(conf_matrix)

print("\n분류 보고서 (Classification Report):")
print(classification_report(y_eval, y_pred))