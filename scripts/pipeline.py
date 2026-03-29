# This file serves as an introductory implementation and is not the final product
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os
import joblib

# Configs
SAMPLING_RATE = 100     # sampling at 100Hz
WINDOW_LENGTH = 200     # 2s at 100Hz
STRIDE = 100            # 50% overlap
NUM_CHANNELS = 6
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRAIN_SPLIT = 0.85
CUTOFF_HZ = 8

SENSOR_COLUMNS = ['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps']

# File Paths
ANNOTATED_BASE_DIR = Path("data/annotated")


def load_all_annotated_sessions(dataset_split):
    split_dir = ANNOTATED_BASE_DIR / dataset_split

    if not split_dir.exists():
        raise ValueError(f"Dataset split folder does not exist: {split_dir}")

    all_sessions = []
    csv_files = sorted(split_dir.glob('*.csv'))

    if not csv_files:
        raise ValueError(f"No CSV files found in {split_dir}")

    for csv_file in csv_files:
        print(f"Loading {csv_file} ...")
        df = pd.read_csv(csv_file, low_memory=False)
        all_sessions.append(df)

    combined = pd.concat(all_sessions, ignore_index=True)
    return combined


def low_pass_filter(data, cutoff_hz, sample_rate_hz, order=4):
    nyquist = sample_rate_hz / 2
    normal_cutoff = cutoff_hz / nyquist
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sosfiltfilt(sos, data)


def apply_filter_to_all_channels(df):
    df_filtered = df.copy()
    for col in SENSOR_COLUMNS:
        df_filtered[col] = low_pass_filter(df[col].values, CUTOFF_HZ, SAMPLING_RATE)
    return df_filtered


def sliding_windows(data, labels, window_length, stride):
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


# def split_dataset(X, y, train_ratio):
#     num_windows = len(X)
#     train_end = int(num_windows * train_ratio)

#     X_train = X[:train_end]
#     y_train = y[:train_end]

#     X_val = X[train_end:]
#     y_val = y[train_end:]

#     return X_train, X_val, y_train, y_val


def normalize(X_train, X_val):
    num_train, window_len, num_channels = X_train.shape
    num_val = X_val.shape[0]

    scaler = StandardScaler()

    X_train_flat = X_train.reshape(-1, num_channels)
    X_val_flat = X_val.reshape(-1, num_channels)

    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(num_train, window_len, num_channels)
    X_val_scaled = scaler.transform(X_val_flat).reshape(num_val, window_len, num_channels)

    return X_train_scaled, X_val_scaled, scaler


def plot_signals(time, signals_dict, title="IMU Signals"):
    num_channels = len(signals_dict)
    fig, axes = plt.subplots(num_channels, 1, figsize=(14, 2.5 * num_channels), sharex=True)
    fig.suptitle(title, fontsize=14)

    if num_channels == 1:
        axes = [axes]

    for ax, (channel_name, series) in zip(axes, signals_dict.items()):
        for label, data in series.items():
            ax.plot(time, data, label=label, linewidth=0.9, alpha=0.8)
        ax.set_ylabel(channel_name, fontsize=8)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()


def plot_raw_vs_filtered(time, raw_df, filtered_df, columns):
    fig, axes = plt.subplots(len(columns), 1, figsize=(14, 2.5 * len(columns)), sharex=True)
    fig.suptitle("Raw vs Filtered IMU Signals", fontsize=14)

    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(time, raw_df[col].values, label='Raw', alpha=0.6, linewidth=0.8)
        ax.plot(time, filtered_df[col].values, label='Filtered', alpha=0.9, linewidth=1.2)
        ax.set_ylabel(col)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_zoomed_segment(time, raw_df, filtered_df, columns, start_s, end_s):
    mask = (time >= start_s) & (time <= end_s)

    fig, axes = plt.subplots(len(columns), 1, figsize=(14, 2.5 * len(columns)), sharex=True)
    fig.suptitle(f"Zoomed Raw vs Filtered ({start_s}s to {end_s}s)", fontsize=14)

    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(time[mask], raw_df.loc[mask, col].values, label='Raw', alpha=0.6, linewidth=0.8)
        ax.plot(time[mask], filtered_df.loc[mask, col].values, label='Filtered', alpha=0.9, linewidth=1.2)
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_windows_zoomed(df, signal_col, window_length, stride, start_time_s, end_time_s):
    time_s = df['time_ms'].values / 1000.0
    signal = df[signal_col].values
    labels = df['label'].values

    mask = (time_s >= start_time_s) & (time_s <= end_time_s)

    plt.figure(figsize=(16, 5))
    plt.plot(time_s[mask], signal[mask], color='black', linewidth=1.2, label=signal_col)

    y_min = np.min(signal[mask])
    y_max = np.max(signal[mask])

    n = len(df)
    for start in range(0, n - window_length + 1, stride):
        end = start + window_length

        x0 = time_s[start]
        x1 = time_s[end - 1]

        # only show windows that overlap the zoomed interval
        if x1 < start_time_s or x0 > end_time_s:
            continue

        window_label = 1 if np.sum(labels[start:end]) > (window_length / 2) else 0
        color = 'green' if window_label == 1 else 'red'

        plt.axvspan(x0, x1, color=color, alpha=0.12)

        # label text near top of plot
        text_x = max(x0, start_time_s) + (min(x1, end_time_s) - max(x0, start_time_s)) / 2
        plt.text(text_x, y_max * 0.95, str(window_label),
                 ha='center', va='top', fontsize=9, color=color, fontweight='bold')

    plt.xlim(start_time_s, end_time_s)
    plt.ylim(y_min - 0.05 * abs(y_min), y_max + 0.1 * abs(y_max))
    plt.title(f"Zoomed Sliding Windows with Labels on {signal_col}")
    plt.xlabel("Time (s)")
    plt.ylabel(signal_col)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_val, y_pred):
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Stitch', 'Stitch'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def preprocessing(dataset_split):
    print(f"Loading {dataset_split} sessions...")
    data_raw = load_all_annotated_sessions(dataset_split)
    print(f"Total samples (before cleaning): {len(data_raw)}")

    required_columns = ['time_ms'] + SENSOR_COLUMNS + ['label']

    # Convert required columns to numeric
    for col in required_columns:
        data_raw[col] = pd.to_numeric(data_raw[col], errors='coerce')

    # Drop rows with invalid or missing values
    bad_rows = data_raw[required_columns].isna().any(axis=1).sum()
    if bad_rows > 0:
        print(f"Dropping {bad_rows} invalid row(s)...")

    data_raw = data_raw.dropna(subset=required_columns).reset_index(drop=True)
    print(f"Total samples (after cleaning): {len(data_raw)}")

    # Ensure labels are integers
    data_raw['label'] = data_raw['label'].astype(int)

    # Filter
    print("Applying low pass filter...")
    data_filtered = apply_filter_to_all_channels(data_raw)

    # Extract signals and labels
    imu_data = data_filtered[SENSOR_COLUMNS].values.astype(np.float32)
    labels = data_filtered['label'].values.astype(int)

    # Sliding windows
    print("Creating sliding windows...")
    X, y = sliding_windows(imu_data, labels, WINDOW_LENGTH, STRIDE)
    print(f"Total windows: {len(X)}")
    print(f"Stitch windows:     {np.sum(y == 1)}")
    print(f"Non-stitch windows: {np.sum(y == 0)}")

    print(f"{dataset_split.capitalize()} preprocessing complete.")

    return data_raw, data_filtered, X, y


def build_model(window_length, num_channels):
    """
    1D-CNN binary classifier for crochet stitch detection.
    Input shape: (window_length, num_channels)
    Output: single probability (0 = no stitch, 1 = stitch)
    """

    model = models.Sequential([

        # Input
        layers.Input(shape=(window_length, num_channels)),

        # First conv block — learn local temporal patterns
        layers.Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # Second conv block — learn higher level patterns
        layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),

        # Third conv block
        layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),

        # Collapse time dimension
        layers.GlobalAveragePooling1D(),

        # Classification head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')

    ])

    return model


def train(model, X_train, y_train, X_val, y_val):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Handle class imbalance
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weights = {0: weights[0], 1: weights[1]}
    print(f"Class weights: {class_weights}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
            callbacks.ModelCheckpoint(
                filepath='models/best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ],
        verbose=1
    )
    return history


def evaluate(model, X_val, y_val):
    y_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_prob >= 0.4).astype(int)

    print("\n" + classification_report(y_val, y_pred, target_names=['Non-Stitch', 'Stitch']))

    cm = confusion_matrix(y_val, y_pred)
    print(f"Confusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

    return y_prob, y_pred





if __name__ == "__main__":
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.makedirs("models", exist_ok=True)

    # time = data_raw['time_ms'].values / 1000

    # raw_signals = {
    #     col: {'Raw': data_raw[col].values}
    #     for col in SENSOR_COLUMNS
    # }

    # filtered_signals = {
    #     col: {'Filtered': data_filtered[col].values}
    #     for col in SENSOR_COLUMNS
    # }

    # plot_signals(time, raw_signals, title="Raw IMU Signals")
    # plot_signals(time, filtered_signals, title="Filtered IMU Signals")
    # plot_raw_vs_filtered(time, data_raw, data_filtered, SENSOR_COLUMNS)
    # plot_zoomed_segment(time, data_raw, data_filtered, SENSOR_COLUMNS, 6000, 6300)
    # plot_windows_zoomed(
    #     data_filtered,
    #     signal_col='ax_g',
    #     window_length=WINDOW_LENGTH,
    #     stride=STRIDE,
    #     start_time_s=6000,
    #     end_time_s=6300
    # )


    # Load and preprocess train split
    train_raw, train_filtered, X_train, y_train = preprocessing("train")

    # Load and preprocess validation split
    val_raw, val_filtered, X_val, y_val = preprocessing("validation")

    # Normalize using training statistics only
    print("Normalizing...")
    X_train, X_val, scaler = normalize(X_train, X_val)

    model = build_model(WINDOW_LENGTH, NUM_CHANNELS)
    model.summary()

    history = train(model, X_train, y_train, X_val, y_val)
    plot_history(history)

    print("\n--- VALIDATION SET EVALUATION ---")
    y_prob, y_pred = evaluate(model, X_val, y_val)

    # Save scaler parameters for ESP32 firmware
    joblib.dump(scaler, 'models/scaler.pkl')
    np.save('models/scaler_mean.npy', scaler.mean_)
    np.save('models/scaler_scale.npy', scaler.scale_)
    print("\nScaler saved to models/")

    # Load best checkpoint before testing
    model = tf.keras.models.load_model('models/best_model.keras')
    print("Loaded best model checkpoint")

    # Load and preprocess test split
    test_raw, test_filtered, X_test, y_test = preprocessing("test")

    # Normalize using training scaler (never refit on test data)
    num_test = X_test.shape[0]
    X_test_flat = X_test.reshape(-1, NUM_CHANNELS)
    X_test_scaled = scaler.transform(X_test_flat).reshape(num_test, WINDOW_LENGTH, NUM_CHANNELS)

    # Evaluate on test set
    print("\n--- TEST SET EVALUATION ---")
    y_prob_test, y_pred_test = evaluate(model, X_test_scaled, y_test)
    # plot_confusion_matrix(y_test, y_pred_test)
