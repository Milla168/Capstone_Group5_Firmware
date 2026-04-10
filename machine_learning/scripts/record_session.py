"""
Simple crochet data recorder with live stitch marking.

Press SPACE to mark the START of a stitch.
Press SPACE again to mark the END of a stitch.
Repeat for each stitch.

Usage:
    python record_session.py               # Uses defaults (auto-detect port, 30 seconds)
    python record_session.py COM15         # Specify port
    python record_session.py COM15 60      # Specify port and duration
"""

import serial
import serial.tools.list_ports
import time
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import RAW_DATA_DIR

try:
    import keyboard
except ImportError:
    print("ERROR: 'keyboard' library not found. Run: pip install keyboard")
    sys.exit(1)

STITCH_KEY = 'space'
DEBOUNCE_S = 0.3                # seconds between accepted keypresses
STITCH_START_OFFSET_MS = -300   # shift stitch start earlier to compensate for reaction time
STITCH_END_OFFSET_MS   =  300   # shift stitch end later to compensate for reaction time


def find_port():
    """Auto-detect ESP32 port."""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if any(keyword in p.description.upper() for keyword in ['CP210', 'CH340', 'USB', 'SERIAL']):
            return p.device
    if ports:
        return ports[0].device
    return None


def record(port, duration):
    """Record sensor data to CSV file with live stitch marking."""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RAW_DATA_DIR / f"crochet_{timestamp}.csv"

    print("=" * 50)
    print("  CROCHET DATA RECORDER")
    print("=" * 50)
    print(f"  Port:        {port}")
    print(f"  Duration:    {duration} seconds")
    print(f"  Output:      {filename}")
    print(f"  Stitch key:  SPACE (press to start, press to stop)")
    print("=" * 50)

    # Connect
    try:
        ser = serial.Serial(port, 115200, timeout=1)
    except serial.SerialException as e:
        print(f"\nERROR: Could not open {port}")
        print(f"  {e}")
        print("\nAvailable ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device} - {p.description}")
        return None

    time.sleep(2)
    ser.reset_input_buffer()

    # Shared state between threads
    stitch_intervals = []               # list of (start_ms, end_ms) in ESP32 time
    current_stitch_start = [None]       # None = not currently in a stitch
    last_esp32_time_ms = [0]            # most recent ESP32 timestamp from serial
    recording_active = [True]
    start_time = time.time()            # PC clock , only used for progress display

    def listen_for_stitches():
        last_press_pc_time = 0

        while recording_active[0]:
            if keyboard.is_pressed(STITCH_KEY):
                now_pc = time.time()

                if now_pc - last_press_pc_time < DEBOUNCE_S:
                    time.sleep(0.01)    # yield CPU while debouncing
                    continue

                last_press_pc_time = now_pc
                press_time_ms = last_esp32_time_ms[0]

                if current_stitch_start[0] is None:
                    current_stitch_start[0] = press_time_ms
                    print(f"  [STITCH START at {press_time_ms}ms]")
                else:
                    stitch_start = current_stitch_start[0]
                    stitch_end = press_time_ms
                    stitch_intervals.append((stitch_start, stitch_end))
                    duration_ms = stitch_end - stitch_start
                    print(f"  [STITCH END   at {press_time_ms}ms, duration: {duration_ms}ms]")
                    current_stitch_start[0] = None

            else:
                time.sleep(0.01)    # key not pressed, yield CPU to main loop

    stitch_thread = threading.Thread(target=listen_for_stitches, daemon=True)
    stitch_thread.start()

    print("\n>>> Recording started")
    print("    SPACE = start stitch | SPACE again = end stitch")
    print()

    header = "time_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,accel_mag,gyro_mag,label"

    sample_count = 0
    last_print = start_time
    lines_buffer = []

    try:
        while True:
            elapsed = time.time() - start_time

            if elapsed >= duration:
                break

            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                # accept lines that start with a digit and have commas
                if ',' in line and line[0].isdigit():
                    lines_buffer.append(line)
                    last_esp32_time_ms[0] = int(float(line.split(',')[0]))
                    sample_count += 1

            if time.time() - last_print >= 1.0:
                remaining = duration - elapsed
                completed = len(stitch_intervals)
                in_progress = "| IN STITCH " if current_stitch_start[0] is not None else ""
                print(f"  {elapsed:.0f}s / {duration}s | {sample_count} samples | {completed} stitches complete {in_progress}| {remaining:.0f}s remaining")
                last_print = time.time()

    except KeyboardInterrupt:
        print("\n\nStopped early by user (Ctrl+C)")

    # If user forgot to close the last stitch, close it at recording end
    if current_stitch_start[0] is not None:
        final_ms = last_esp32_time_ms[0]
        stitch_intervals.append((current_stitch_start[0], final_ms))
        print(f"  [WARNING: Stitch was still open at end of recording, closed at {final_ms}ms]")

    recording_active[0] = False
    ser.close()
    elapsed_total = time.time() - start_time

    # Write CSV with labels
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(header + '\n')

        for raw_line in lines_buffer:
            sample_ms = int(float(raw_line.split(',')[0]))

            label = 0
            for stitch_start, stitch_end in stitch_intervals:
                adjusted_start = stitch_start + STITCH_START_OFFSET_MS
                adjusted_end   = stitch_end   + STITCH_END_OFFSET_MS
                if adjusted_start <= sample_ms <= adjusted_end:
                    label = 1
                    break

            f.write(f"{raw_line},{label}\n")

    print()
    print("=" * 50)
    print("  RECORDING COMPLETE")
    print("=" * 50)
    print(f"  Duration:          {elapsed_total:.1f} seconds")
    print(f"  Samples:           {sample_count}")
    print(f"  Stitches marked:   {len(stitch_intervals)}")
    if elapsed_total > 0:
        print(f"  Sample rate:       {sample_count/elapsed_total:.1f} Hz")
    print(f"  File size:         {os.path.getsize(filename):,} bytes")
    print(f"  Saved to:          {os.path.abspath(filename)}")
    print("=" * 50)
    print()
    print("  Stitch intervals (ms):")
    for i, (start, end) in enumerate(stitch_intervals, 1):
        adjusted_start = start + STITCH_START_OFFSET_MS
        adjusted_end   = end   + STITCH_END_OFFSET_MS
        print(f"    Stitch {i:>2}: {start}ms → {end}ms  (raw) | {adjusted_start}ms → {adjusted_end}ms  (adjusted) | {end - start}ms duration")
    print()

    return filename


def main():
    port = None
    duration = 30

    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print(__doc__)
            return
        port = sys.argv[1]

    if len(sys.argv) > 2:
        try:
            duration = int(sys.argv[2])
        except ValueError:
            print(f"ERROR: Duration must be a number (got '{sys.argv[2]}')")
            return

    if port is None:
        port = find_port()
        if port is None:
            print("ERROR: No serial port found")
            print("Specify port manually: python simple_record.py COM3")
            return
        print(f"Auto-detected port: {port}")

    record(port, duration)


if __name__ == '__main__':
    main()