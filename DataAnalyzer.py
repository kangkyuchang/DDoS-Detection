import pandas as pd
import numpy as np
import winsplit as ws
from jedi.api import file_name


def separate_save(file_name):
    df = pd.read_csv(f"dataset/03-11/{file_name}.csv")
    df.columns = df.columns.str.strip()
    normal = df[df["Label"] == "BENIGN"]
    print(normal["Label"].value_counts())
    normal.to_csv(f"dataset/separate/BENIGN_{file_name}.csv", index=False)

df = pd.read_csv("dataset/DDoSDataset.csv")
drop_column = ["Unnamed: 0", "Flow ID", "Destination IP", "Timestamp", "SimillarHTTP"]
df = df.drop(columns=drop_column)
df.replace(np.inf, np.nan, inplace=True)
df = df.fillna(df.max())

ws.spilt_by_label(df)

# df = pd.read_csv("dataset/separate/ATTACK_UDP.csv")
# df.columns = df.columns.str.strip()

# df = df.sample(n=30000, random_state=42)
# df.to_csv(f"dataset/separate/ATTACK_SAMPLE_UDP.csv", index=False)


# LDAP = pd.read_csv("dataset/separate/BENIGN_LDAP.csv")
# MSSQL = pd.read_csv("dataset/separate/BENIGN_MSSQL.csv")
# NetBIOS = pd.read_csv("dataset/separate/BENIGN_NetBIOS.csv")
# Portmap = pd.read_csv("dataset/separate/BENIGN_Portmap.csv")
# Syn = pd.read_csv("dataset/separate/BENIGN_Syn.csv")
# UDP = pd.read_csv("dataset/separate/BENIGN_UDP.csv")
# UDPLag = pd.read_csv("dataset/separate/BENIGN_UDPLag.csv")
#
# data = pd.concat([LDAP, MSSQL, NetBIOS, Portmap, Syn, UDP, UDPLag], axis=0, ignore_index=True)
# data["Timestamp"] = pd.to_datetime(data["Timestamp"])
# data = data.sort_values("Timestamp").reset_index(drop=True)
#
# data.to_csv("dataset/BENIGN.csv", index=False)