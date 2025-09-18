import xmlrpc.server

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
import numpy as np
from sklearn.impute import SimpleImputer

def create_windows(data, labels, window_size):
    num_samples = data.shape[0] - window_size + 1
    X = np.array([data[i:i+window_size] for i in range(num_samples)])
    y = np.array([1 if 1 in labels[i:i+window_size] else 0 for i in range(num_samples)])
    return X, y

benign = pd.read_csv("separate/Benign.csv")
syn = pd.read_csv("separate/Syn.csv")
udp = pd.read_csv("separate/UDP.csv")

df = pd.concat([benign, syn, udp], axis=0, ignore_index=True)
df = df.drop("SimillarHTTP", axis=1)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp")

labelMapping = {"BENIGN" : 0, "Syn": 1, "UDP": 1}
df["Label"] = df["Label"].map(labelMapping)

df = df.replace([np.inf, -np.inf], np.nan)

X = df.iloc[:, 9: -1]
y = df["Label"]
print(y.shape)

# X_train, y_train = create_windows(X, y, 100)

# cutoff = int(len(df) * 0.8)
# X_train = X[:cutoff].copy()
# y_train = y[:cutoff]
#
# X_test = X[cutoff:].copy()
# y_test = y[cutoff:]

imputer = SimpleImputer(strategy="mean")
X[["Flow Bytes/s", "Flow Packets/s"]] = imputer.fit_transform(X[["Flow Bytes/s", "Flow Packets/s"]])
# X_test[["Flow Bytes/s", "Flow Packets/s"]] = imputer.transform(X_test[["Flow Bytes/s", "Flow Packets/s"]])

rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
selector = RFECV(rf, step=1, cv=5, scoring='accuracy')
selector.fit(X, y)

# y_pred = selector.predict(X_test)
#
# print("정확도:", accuracy_score(y_test, y_pred))
# print("분류 리포트:\n", classification_report(y_test, y_pred))

pd.set_option('display.max_rows', None)

X_train_selected = selector.transform(X)
selected_features = X.columns[selector.support_]
print(selected_features)

feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
feature_importance = feature_importance.sort_values(ascending=False)
important_features = feature_importance[feature_importance > 0].index
print(important_features)
feature_importance.plot(kind="barh")
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# ['Total Backward Packets', 'Total Length of Fwd Packets',
#        'Fwd Packet Length Max', 'Fwd Packet Length Min',
#        'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Min Packet Length',
#        'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
#        'Packet Length Variance', 'ACK Flag Count', 'URG Flag Count',
#        'CWE Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size',
#        'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Init_Win_bytes_forward',
#        'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'Inbound']