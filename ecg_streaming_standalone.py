import collections
import threading
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sifi_bridge_py as sbp


# ---------------------------------------------------------------------------
# Live plot configuration.
#
#   IMU_FS         IMU sampling rate in Hz. Must match configure_imu(fs=...)
#                  so the time axis is correct.
#   WINDOW_SEC     How many seconds of history to keep on screen. The plot
#                  scrolls left as new samples arrive.
#   FRAME_INTERVAL How often (in ms) matplotlib redraws the plot. The data
#                  itself arrives much faster than this on the BLE side —
#                  we just don't need to redraw at the BLE rate.
# ---------------------------------------------------------------------------
ECG_FS = 500
WINDOW_SEC = 10
FRAME_INTERVAL = 50  # ms


# ---------------------------------------------------------------------------
# Shared state between the acquisition thread (which receives BLE packets)
# and the plotting thread (matplotlib's main loop).
#
# A deque with a bounded maxlen automatically discards the oldest sample
# when full, so we always hold exactly WINDOW_SEC seconds of history with
# no manual trimming. Appends and iteration are atomic enough under the
# GIL for a tutorial-grade plot.
# ---------------------------------------------------------------------------
norm_buffer = collections.deque(maxlen=ECG_FS * WINDOW_SEC)
stop_event = threading.Event()


def acquisition_loop(sb):
    """Background thread: pull ECG packets and append the signal."""
    while not stop_event.is_set():
        packet = sb.get_data(timeout=0.5)
        if not packet or packet.get('packet_type') != 'ecg':
            continue

        # ECG packets carry the raw ECG signal.
        # We append the signal to the buffer.
        data = packet['data']
        ecg_samples = np.asarray(data["ecg"], dtype=float).reshape(-1)
        norm_buffer.extend(float(sample) for sample in ecg_samples)

# ---------------------------------------------------------------------------
# Connect, configure, start streaming.
# ---------------------------------------------------------------------------
sb = sbp.SifiBridge()
sb.connect()

# Restrict streaming to the IMU only so the BLE link isn't shared with
# sensors we don't plot. configure_sensors() is the on/off switch;
# configure_imu() then sets the IMU-specific parameters. accel_range=8
sb.configure_sensors(ecg=True)
sb.configure_ecg(fs=ECG_FS)

sb.start()
time.sleep(1.0)
sb.clear_data_buffer()

# Spawn the acquisition thread. It runs until stop_event is set in the
# cleanup block below.
acq_thread = threading.Thread(target=acquisition_loop, args=(sb,), name="imu-acq")
acq_thread.start()

# ---------------------------------------------------------------------------
# Matplotlib live plot. FuncAnimation calls update() every FRAME_INTERVAL
# ms; update() snapshots the buffer and redraws the line. The window
# scrolls left as new samples arrive because xlim is anchored to the
# right edge of the buffer.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
line, = ax.plot([], [], color='tab:purple', linewidth=0.9)
ax.set_xlim(0, WINDOW_SEC)
ax.set_ylim(0, 3)
ax.set_xlabel("Time (s, trailing window)")
ax.set_ylabel("|a|")
ax.set_title(f"Live ECG signal — last {WINDOW_SEC} s")
ax.grid(True, alpha=0.3)


def update(_frame):
    samples = np.asarray(norm_buffer, dtype=float).reshape(-1)
    n = samples.size
    if n == 0:
        return (line,)

    # Snapshot the deque so the line draw sees a stable view even if the
    # acquisition thread keeps appending behind us.
    t = [(i - (n - 1)) / ECG_FS + WINDOW_SEC for i in range(n)]
    line.set_data(t, samples)

    # Auto-rescale Y if motion goes outside the default range.
    smin = float(samples.min())
    smax = float(samples.max())
    if smin == smax:
        pad = max(abs(smax) * 0.1, 1.0)
    else:
        pad = (smax - smin) * 0.1
    ax.set_ylim(smin - pad, smax + pad)
    return (line,)


anim = animation.FuncAnimation(
    fig, update, interval=FRAME_INTERVAL, blit=True, cache_frame_data=False)

print("Live IMU norm plot. Close the window to stop.")
try:
    plt.show()  # blocks until the window is closed
finally:
    # Clean shutdown: tell the acquisition thread to exit, then tear
    # down the BLE link.
    stop_event.set()
    acq_thread.join(timeout=2.0)
    sb.stop()
    sb.disconnect()
    time.sleep(0.5)
    print("Disconnected.")