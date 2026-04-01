# src/preprocess.py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from configs.config import SENSOR_COLUMNS, CUTOFF_HZ, SAMPLING_RATE, WINDOW_LENGTH, STRIDE


def load_sessions(split_dir):
    """
    Load the specified .csv dataset files

        Args:
            split_dir (string): The target dataset folder to be loaded (train, validation, test)

        Returns:
            The combination of all crochet sessions in the specified dataset folder

        Raises:
            ValueError: If csv files found in specified folder.
    """

    all_sessions = []
    csv_files = sorted(split_dir.glob('*.csv'))

    if not csv_files:
        raise ValueError(f"No CSV files found in {split_dir}")

    for csv_file in csv_files:
        print(f"  Loading {csv_file.name} ...")
        df = pd.read_csv(csv_file, low_memory=False)
        all_sessions.append(df)

    combined_dataset = pd.concat(all_sessions, ignore_index=True)

    return combined_dataset


def low_pass_filter(data, cutoff_hz=CUTOFF_HZ, sample_rate_hz=SAMPLING_RATE, order=4):
    """
    Applies low pass filter to sensor data

        Args:
            data ():
            cutoff_hz ():
            sample_rate_hz (): 
            order (int):

        Returns:
            

        Raises:
            ValueError: If csv files found in specified folder.
    """

    nyquist = sample_rate_hz / 2
    normal_cutoff = cutoff_hz / nyquist
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    data_filtered = sosfiltfilt(sos, data)

    return data_filtered


def apply_filter(df):
    df_filtered = df.copy()
    for col in SENSOR_COLUMNS:
        df_filtered[col] = low_pass_filter(df[col].values, CUTOFF_HZ, SAMPLING_RATE)

    return df_filtered


def sliding_windows(data, labels, window_length=WINDOW_LENGTH, stride=STRIDE):
    X = []
    y = []

    num_samples = len(data)

    for start in range(0, num_samples - window_length + 1, stride):
        end = start + window_length

        window_data = data[start:end]
        window_labels = labels[start:end]

        label = 1 if np.sum(window_labels) > (window_length / 2) else 0

        X.append(window_data)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


def normalize(X_train, X_val, X_test=None):
    num_train, window_len, num_channels = X_train.shape

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_channels)).reshape(X_train.shape)

    X_val_scaled = scaler.transform(X_val.reshape(-1, num_channels)).reshape(X_val.shape)

    if X_test is not None:
        X_test_scaled = scaler.transform( X_test.reshape(-1, num_channels)).reshape(X_test.shape)
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler

    return X_train_scaled, X_val_scaled, scaler


def clean(df):
    required = ['time_ms'] + SENSOR_COLUMNS + ['label']
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    bad = df[required].isna().any(axis=1).sum()
    if bad > 0:
        print(f"  Dropping {bad} invalid rows")

    df = df.dropna(subset=required).reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    return df


def preprocess_split(split_dir):
    print(f"Loading {split_dir.name}...")
    df_raw = load_sessions(split_dir)
    print(f"  Samples before cleaning: {len(df_raw)}")

    df_raw = clean(df_raw)
    print(f"  Samples after cleaning:  {len(df_raw)}")

    df_filtered = apply_filter(df_raw)

    imu_data = df_filtered[SENSOR_COLUMNS].values.astype(np.float32)
    labels   = df_filtered['label'].values.astype(int)

    X, y = sliding_windows(imu_data, labels)
    print(f"  Windows: {len(X)} | Stitch: {np.sum(y==1)} | Non-stitch: {np.sum(y==0)}")

    return df_raw, df_filtered, X, y