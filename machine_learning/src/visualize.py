# src/visualize.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
        ax.plot(time, raw_df[col].values,      label='Raw',      alpha=0.6, linewidth=0.8)
        ax.plot(time, filtered_df[col].values, label='Filtered', alpha=0.9, linewidth=1.2)
        ax.set_ylabel(col)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_signal_segment(time, raw_df, filtered_df, columns, start_s, end_s):
    mask = (time >= start_s) & (time <= end_s)

    fig, axes = plt.subplots(len(columns), 1, figsize=(14, 2.5 * len(columns)), sharex=True)
    fig.suptitle(f"Zoomed Raw vs Filtered ({start_s}s to {end_s}s)", fontsize=14)

    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(time[mask], raw_df.loc[mask, col].values,      label='Raw',      alpha=0.6, linewidth=0.8)
        ax.plot(time[mask], filtered_df.loc[mask, col].values, label='Filtered', alpha=0.9, linewidth=1.2)
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_windows_segment(df, signal_col, window_length, stride, start_time_s, end_time_s):
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

        if x1 < start_time_s or x0 > end_time_s:
            continue

        window_label = 1 if np.sum(labels[start:end]) > (window_length / 2) else 0
        color = 'green' if window_label == 1 else 'red'

        plt.axvspan(x0, x1, color=color, alpha=0.12)

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


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Non-Stitch', 'Stitch'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.show()


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'],     label='Train')
    ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['accuracy'],     label='Train')
    ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()