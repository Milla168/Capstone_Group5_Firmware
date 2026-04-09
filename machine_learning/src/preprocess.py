# src/preprocess.py
"""
This module handles data preprocessing for the crochet stitch detection pipeline.
It provides functions for:

1. Loading CSV session files from dataset directories
2. Cleaning and validating raw sensor data
3. Applying low-pass filtering to remove high-frequency noise
4. Segmenting continuous data into sliding windows
5. Normalizing features using StandardScaler

Functions:
    load_sessions: Load and combine CSV files from a directory
    low_pass_filter: Apply Butterworth low-pass filter to signal data
    apply_filter: Apply filtering to all sensor columns in a dataframe
    sliding_windows: Segment data into overlapping windows with majority-vote labels
    normalize: Standardize features using training data statistics
    clean: Remove invalid rows and convert columns to proper dtypes
    preprocess_split: Complete preprocessing pipeline for a dataset split

"""


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
            ValueError: If no csv files found in specified folder.
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



# def low_pass_filter(data, cutoff_hz=CUTOFF_HZ, sample_rate_hz=SAMPLING_RATE, order=4):
#     """
#     Applies low pass filter to sensor data to remove high-frequency noise

#         Args:
#             data (np.ndarray): Raw sensor signal data to be filtered
#             cutoff_hz (float): Cutoff frequency in Hz for the low pass filter
#             sample_rate_hz (float): Sampling rate of the sensor data in Hz
#             order (int): Order of the Butterworth filter

#         Returns:
#             np.ndarray: Filtered sensor data with high-frequency noise removed
#     """

#     nyquist = sample_rate_hz / 2
#     normal_cutoff = cutoff_hz / nyquist
#     sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
#     data_filtered = sosfiltfilt(sos, data)

#     return data_filtered



# def apply_filter(df):
#     """
#     Applies low pass filter to all sensor columns in the dataframe

#         Args:
#             df (pd.DataFrame): Dataframe containing raw sensor data

#         Returns:
#             pd.DataFrame: Copy of dataframe with filtered sensor columns
#     """
#     df_filtered = df.copy()
#     for col in SENSOR_COLUMNS:
#         df_filtered[col] = low_pass_filter(df[col].values, CUTOFF_HZ, SAMPLING_RATE)

#     return df_filtered



def sliding_windows(data, labels, window_length=WINDOW_LENGTH, stride=STRIDE):
    """
    Segments continuous sensor data into fixed-length overlapping windows

        Args:
            data (np.ndarray): Sensor data array of shape (num_samples, num_channels)
            labels (np.ndarray): Binary label array of shape (num_samples,)
            window_length (int): Number of samples per window
            stride (int): Number of samples to shift between consecutive windows

        Returns:
            tuple: (X, y) where X is windowed data of shape (num_windows, window_length, num_channels)
                and y is binary labels of shape (num_windows,) based on majority voting
    """

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



# def normalize(X_train, X_val, X_test=None):
#     """
#     Normalizes sensor data using StandardScaler fitted on training data

#         Args:
#             X_train (np.ndarray): Training data of shape (num_windows, window_length, num_channels)
#             X_val (np.ndarray): Validation data of shape (num_windows, window_length, num_channels)
#             X_test (np.ndarray, optional): Test data of shape (num_windows, window_length, num_channels)

#         Returns:
#             tuple: Scaled arrays (X_train_scaled, X_val_scaled, [X_test_scaled], scaler)
#     """

#     num_train, window_len, num_channels = X_train.shape

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_channels)).reshape(X_train.shape)

#     X_val_scaled = scaler.transform(X_val.reshape(-1, num_channels)).reshape(X_val.shape)

#     if X_test is not None:
#         X_test_scaled = scaler.transform( X_test.reshape(-1, num_channels)).reshape(X_test.shape)
#         return X_train_scaled, X_val_scaled, X_test_scaled, scaler

#     return X_train_scaled, X_val_scaled, scaler



def clean(df):
    """
    Cleans dataframe by converting columns to numeric and removing invalid rows

        Args:
            df (pd.DataFrame): Raw dataframe containing sensor data and labels

        Returns:
            pd.DataFrame: Cleaned dataframe with invalid rows removed and proper dtypes
    """

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
    """
    Complete preprocessing pipeline for a dataset split

        Args:
            split_dir (Path): Path to directory containing CSV files for this split

        Returns:
            tuple: (df_raw, df_filtered, X, y) containing raw dataframe, filtered dataframe,
                windowed feature array, and windowed label array
    """

    print(f"Loading {split_dir.name}...")
    df_raw = load_sessions(split_dir)
    print(f"  Samples before cleaning: {len(df_raw)}")

    df_raw = clean(df_raw)
    print(f"  Samples after cleaning:  {len(df_raw)}")

    # df_filtered = apply_filter(df_raw)

    imu_data = df_raw[SENSOR_COLUMNS].values.astype(np.float32)
    labels   = df_raw['label'].values.astype(int)

    X, y = sliding_windows(imu_data, labels)
    print(f"  Windows: {len(X)} | Stitch: {np.sum(y==1)} | Non-stitch: {np.sum(y==0)}")

    return df_raw, df_raw, X, y