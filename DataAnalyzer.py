import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('separate/Benign.csv')

# benign = pd.read_csv("separate/Benign(M).csv")
# syn = pd.read_csv("separate/Syn.csv")
# udp = pd.read_csv("separate/UDP.csv")
#
# df = pd.concat([benign, syn, udp], axis=0, ignore_index=True)

# df['minute'] = df['Timestamp'].dt.floor('min')
# count_by_minute = df.groupby('minute').size()

# with open('output.csv', 'w', encoding='utf-8') as f:
#     f.write(df[["Timestamp", "Label"]].to_csv(index=False))

# LDAP = pd.read_csv("separate/Benign_LDAP.csv")
# MSSQL = pd.read_csv("separate/Benign_MSSQL.csv")
# NetBIOS = pd.read_csv("separate/Benign_NetBIOS.csv")
# Portmap = pd.read_csv("separate/Benign_Portmap.csv")
# Syn = pd.read_csv("separate/Benign_Syn.csv")
# UDP = pd.read_csv("separate/Benign_UDP.csv")
# UDPLag = pd.read_csv("separate/Benign_UDPLag.csv")
#
# data = pd.concat([LDAP, MSSQL, NetBIOS, Portmap, Syn, UDP, UDPLag], axis=0, ignore_index=True)
# data["Timestamp"] = pd.to_datetime(data["Timestamp"])
# data = data.sort_values("Timestamp").reset_index(drop=True)
#
# data.to_csv("separate/Benign.csv", index=False)

# features1 = ['Total Backward Packets', 'Total Length of Fwd Packets',
#        'SYN Flag Count', 'Fwd Packet Length Max', 'Fwd Packet Length Min',
#        'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Min Packet Length',
#        'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
#        'Packet Length Variance', 'ACK Flag Count', 'URG Flag Count',
#        'CWE Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size',
#        'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Init_Win_bytes_forward',
#        'Init_Win_bytes_backward', 'act_data_pkt_fwd']
#
features2 = ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
       'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'SYN Flag Count',
       'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Fwd PSH Flags',
       'Min Packet Length', 'RST Flag Count', 'ACK Flag Count',
       'URG Flag Count', 'CWE Flag Count', 'Average Packet Size',
       'Avg Bwd Segment Size', 'Subflow Fwd Bytes', 'Subflow Bwd Bytes',
       'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd']

# features3 = ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
#        'Fwd Packet Length Min', 'Bwd Packet Length Max', 'SYN Flag Count',
#        'Bwd Packet Length Mean', 'Fwd PSH Flags', 'Bwd Packets/s',
#        'Min Packet Length', 'Packet Length Mean', 'RST Flag Count',
#        'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
#        'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
#        'Subflow Fwd Bytes', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
#        'Init_Win_bytes_backward', 'act_data_pkt_fwd']

features4 = ['Total Length of Fwd Packets', 'Total Length of Bwd Packets',
       'Fwd Packet Length Min', 'Fwd Packet Length Mean',
       'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Fwd PSH Flags',
       'Bwd Packets/s', 'Min Packet Length', 'Packet Length Mean',
       'RST Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count',
       'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
       'Subflow Fwd Bytes', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward',
       'Init_Win_bytes_backward']
#
result = list(set(features2) & set(features4))
print(len(result))
