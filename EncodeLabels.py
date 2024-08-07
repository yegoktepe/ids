# -*- coding: utf-8 -*-
"""
Created on Fri Feb 2 16:05:15 2024

@author: yegoktepe
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

# Load the KDD'99 dataset
column_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'value'
]

# Note: Adjust the path to the 'kddcup.data' file as needed
data = pd.read_csv('NSLKDD.data', header=None, names=column_names)

data['label'] = data['label'].str.replace('normal.', '0')

replacement_dict = {
    'buffer_overflow.': '1',
    'loadmodule.': '1',
    'perl.': '1',
    'neptune.': '1',
    'back.': '1',
    'land.': '1',
    'pod.': '1',
    'smurf.': '1',
    'teardrop.': '1',
    'apache2.': '1',
    'udpstorm.': '1',
    'processtable.': '1',
    'satan.': '1',
    'ipsweep.': '1',
    'portsweep.': '1',
    'nmap.': '1',
    'mscan.': '1',
    'saint.': '1',
    'warezclient.': '1',
    'warezmaster.': '1',
    'ftp_write.': '1',
    'guess_passwd.': '1',
    'imap.': '1',
    'multihop.': '1',
    'phf.': '1',
    'spy.': '1',
    'warez.': '1',
    'xlock.': '1',
    'xsnoop.': '1',
    'rootkit.': '1',
    'ps.': '1',
    'xterm.': '1',
    'sqlattack.': '1'
    # Ek metin eşlemelerini buraya ekleyebilirsiniz
}

# Her satır için döngü ile değiştirme
for index, row in data.iterrows():
    for old_text, new_text in replacement_dict.items():
        if old_text in row['label']:
            data.at[index, 'label'] = row['label'].replace(old_text, new_text)
