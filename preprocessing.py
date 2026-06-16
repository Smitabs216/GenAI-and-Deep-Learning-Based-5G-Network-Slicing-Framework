# ============================================================
#  preprocessing.py
#  Loads the 5G dataset, builds per-slice time-series
#  sequences for LSTM training.
#
#  FIX: PLR is log10-transformed before training so the
#       LSTM learns on a compact, learnable scale.
#       Predictions are converted back via 10^x and clipped.
#
#  Features (X):
#    Throughput_Mbps, Packet_Delay_Budget_ms, PLR_log,
#    Network_Load_Percent, SNR_dB, Num_UEs
#  Targets (y):
#    Throughput_Mbps, Packet_Delay_Budget_ms, PLR_log
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import SLICES, SEQ_LEN, DATASET_PATH

FEATURE_COLS = [
    "Throughput_Mbps",
    "Packet_Delay_Budget_ms",
    "PLR_log",
    "Network_Load_Percent",
    "SNR_dB",
    "Num_UEs",
]
TARGET_COLS = [
    "Throughput_Mbps",
    "Packet_Delay_Budget_ms",
    "PLR_log",
]
TARGET_COLS_REAL = [
    "Throughput_Mbps",
    "Packet_Delay_Budget_ms",
    "Packet_Loss_Rate",
]


def load_dataset(path=DATASET_PATH):
    df = pd.read_csv(path)
    df["Packet_Loss_Rate"] = df["Packet_Loss_Rate"].astype(float).clip(lower=1e-9)
    df["PLR_log"] = np.log10(df["Packet_Loss_Rate"])
    return df


def build_sequences(df, slice_name, seq_len=SEQ_LEN):
    sub = df[df["Slice_Type"] == slice_name].reset_index(drop=True)
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(sub[FEATURE_COLS].values)
    y_scaled = scaler_y.fit_transform(sub[TARGET_COLS].values)
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - seq_len):
        X_seq.append(X_scaled[i: i + seq_len])
        y_seq.append(y_scaled[i + seq_len])
    return np.array(X_seq), np.array(y_seq), scaler_X, scaler_y


def inverse_transform_predictions(y_scaled, scaler_y):
    """Inverse-transform: PLR column converted 10^x and clipped >= 0."""
    y_log  = scaler_y.inverse_transform(y_scaled)
    y_real = y_log.copy()
    y_real[:, 2] = np.clip(10 ** y_log[:, 2], 0.0, 1.0)
    return y_real


def train_test_split_seq(X, y, test_ratio=0.2):
    split = int(len(X) * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


def prepare_all_slices(path=DATASET_PATH):
    df = load_dataset(path)
    data = {}
    for slc in SLICES:
        X, y, sx, sy = build_sequences(df, slc)
        Xtr, Xte, ytr, yte = train_test_split_seq(X, y)
        data[slc] = {
            "X_train": Xtr, "X_test": Xte,
            "y_train": ytr, "y_test": yte,
            "scaler_X": sx, "scaler_y": sy,
            "raw": df[df["Slice_Type"] == slc].reset_index(drop=True),
        }
        print(f"  [{slc}]  train={len(Xtr)}  test={len(Xte)}  "
              f"features={X.shape[2]}  targets={y.shape[1]}")
    return data, df
