import keras
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import winsplit as ws
from keras import models

df = pd.read_csv("../dataset/separate/ATTACK_UDP.csv")
df.columns = df.columns.str.strip()
# df = df[df["Label"] == "BENIGN"]
df = df[:80000]
drop_column = ['Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Protocol', 'Timestamp', "SimillarHTTP", 'Inbound']
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)
df.replace(np.inf, np.nan, inplace=True)
df["Label"] = df["Label"].map(lambda x: 0 if x == "BENIGN" else 1)
df.drop(columns=drop_column, inplace=True)

X = df.iloc[:, :-1]
y = df.iloc[:, -1].values

with open('autoencoder/max_values_autoencoder.pkl', 'rb') as f:
    max_values = pickle.load(f)

X = X.fillna(max_values)
X = X.values

with open('autoencoder/scaler_autoencoder.pkl', 'rb') as f:
    scaler = pickle.load(f)

X = scaler.transform(X)

with open('autoencoder/scaler_minmax_autoencoder.pkl', 'rb') as f:
    scaler_minmax = pickle.load(f)

X = scaler_minmax.transform(X)

window_size = 100
X, y = ws.create_windows(X, y, window_size, 40)
y = y.reshape(-1, 1)

# model = models.load_model("DDoS_detection_autoencoder.h5")

model = models.load_model("autoencoder/DDoS_detection_autoencoder.h5",
                          custom_objects={'mse': keras.metrics.mean_squared_error})

y_pred_proba = model.predict(X)
errors = np.mean(np.square(X - y_pred_proba), axis=(1,2))
threshold = 0.0118

print(f"설정된 임계치: {threshold}")

y_pred = (errors > threshold).astype(int)

# y_pred_proba = model.predict(X)
# y_pred = (y_pred_proba > 0.5).astype(int)
#
print("### 분류 평가 지표 ###")

accuracy = accuracy_score(y, y_pred)
print(f"정확도 (Accuracy): {accuracy:.4f}")

precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}")
print(f"F1-점수 (F1-Score): {f1:.4f}")

conf_matrix = confusion_matrix(y, y_pred)
print("\n혼동 행렬 (Confusion Matrix):")
print(conf_matrix)
#
# print("\n분류 보고서 (Classification Report):")
# print(classification_report(y, y_pred))
