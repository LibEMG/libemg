import asyncio
import struct
import numpy as np
from multiprocessing import Process, Array, Value, Event, freeze_support
from bleak import BleakClient, BleakScanner
import time

# UUIDs
CONTROL_CHAR_UUID = "d5060401-a904-deb9-4748-2c7f4a124842"
EMG_CHAR_UUIDS = [
    "d5060105-a904-deb9-4748-2c7f4a124842",
    "d5060205-a904-deb9-4748-2c7f4a124842",
    "d5060305-a904-deb9-4748-2c7f4a124842",
    "d5060405-a904-deb9-4748-2c7f4a124842",
]

INITIAL_COMMANDS = [
    bytes([0x09, 0x01, 0x01]),
    bytes([0x01, 0x03, 0x02, 0x00, 0x00]),
    bytes([0x03, 0x01, 0x03]),
]

class MinimalMyoBLE(Process):
    def __init__(self, mac_address, channels=8, samples=2):
        super().__init__(daemon=True)
        self.mac = mac_address
        self.shared_data = Array('f', channels * samples)
        self.shared_flag = Value('b', False)
        self.signal = Event()
        self.connected = Value('b', False)

    def run(self):
        asyncio.run(self.start_stream())

    async def start_stream(self):
        try:
            async with BleakClient(self.mac, timeout=20) as client:
                print(f"[{self.mac}] Connected")
                await asyncio.sleep(2)

                for cmd in INITIAL_COMMANDS:
                    await client.write_gatt_char(CONTROL_CHAR_UUID, cmd)
                    await asyncio.sleep(0.05)

                await asyncio.sleep(2)

                for uuid in EMG_CHAR_UUIDS:
                    await client.start_notify(uuid, self._make_handler())

                self.connected.value = True
                print(f"[{self.mac}] Streaming started")

                while not self.signal.is_set():
                    await asyncio.sleep(0.05)

                print(f"[{self.mac}] Streaming stopped")

        except Exception as e:
            print(f"[{self.mac} ERROR] {e}")

    def _make_handler(self):
        def handler(sender, data: bytearray):
            emg_vector = [struct.unpack("b", bytes([b]))[0] / 128.0 for b in data]
            for i in range(min(len(self.shared_data), len(emg_vector))):
                self.shared_data[i] = emg_vector[i]
            self.shared_flag.value = True
        return handler

    def stop(self):
        self.signal.set()
        self.join()

    def read(self):
        if self.shared_flag.value:
            data = list(self.shared_data)  # 16 floats
            self.shared_flag.value = False
            return np.array([data[:8], data[8:]], dtype=np.float32)
        return None

    def is_connected(self):
        return self.connected.value


class MinimalMyosBLEStreamer:
    def __init__(self, mac_addresses):
        self.devices = []
        self.mac_addresses = mac_addresses

    def start_streaming(self):
        print("Connecting to Myos sequentially...")

        for i, mac in enumerate(self.mac_addresses):
            print(f"🔌 Starting Myo {i + 1} at {mac}")
            dev = self._connect_with_retry(mac, i)
            self.devices.append(dev)

        print("✅ All Myos connected and streaming.")

    def _connect_with_retry(self, mac, index, max_retries=5, retry_delay=5):
        attempt = 0
        while attempt < max_retries:
            dev = MinimalMyoBLE(mac)
            dev.start()

            print(f"⏳ Waiting for Myo {index + 1} to connect (attempt {attempt + 1})...")
            timeout = 40  # seconds
            waited = 0
            while waited < timeout:
                if dev.is_connected():
                    print(f"✅ Myo {index + 1} connected!")
                    return dev
                time.sleep(0.2)
                waited += 0.2

            print(f"⚠️ Myo {index + 1} failed to connect. Retrying...")
            dev.terminate()
            dev.join()
            attempt += 1
            time.sleep(retry_delay)

        raise RuntimeError(f"❌ Could not connect to Myo {index + 1} at {mac} after {max_retries} retries.")

    def read(self):
        samples = []
        for dev in self.devices:
            data = dev.read()
            if data is None:
                return None
            samples.append(data)

        return np.hstack(samples)  # shape: (2, 8 * num_myos)

    def stop(self):
        for dev in self.devices:
            dev.stop()


# Optional BLE scanner
async def scan_ble_devices(timeout=10):
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=timeout)
    if not devices:
        print("No BLE devices found.")
    else:
        print("Found BLE devices:")
        for d in devices:
            name = d.name or "Unknown"
            print(f"Name: {name}, Address: {d.address}, RSSI: {d.rssi}")



import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import time
import threading

def main():
    mac_addresses = [
        "C2:44:51:6A:0C:4B",  # Myo 1
        # "FD:5D:41:18:6E:F8",  # Myo 2
        # Add more if needed
    ]

    # Start the streamer
    streamer = MinimalMyosBLEStreamer(mac_addresses)
    streamer.start_streaming()

    num_myos = len(mac_addresses)
    num_channels = 8 * num_myos
    samples_per_read = 2
    sampling_rate = 200  # Hz
    buffer_len = sampling_rate  # Show last 1 second

    # Prepare data buffers
    data_buffers = [deque([0.0] * buffer_len, maxlen=buffer_len) for _ in range(num_channels)]

    # Set up matplotlib figure
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], lw=1)[0] for _ in range(num_channels)]
    ax.set_xlim(0, buffer_len)
    ax.set_ylim(-1, num_channels * 2)
    ax.set_yticks([])
    ax.set_xlabel("Time (samples)")
    ax.set_title("Live EMG from Myo Armbands (200 Hz)")

    x = np.arange(buffer_len)

    # --- Background thread to collect EMG data ---
    def collect_data():
        while True:
            emg_block = streamer.read()  # shape (2, 8 * num_myos)
            if emg_block is not None:
                for ch in range(num_channels):
                    offset = ch * 2.0
                    data_buffers[ch].append(emg_block[0][ch] + offset)
                    data_buffers[ch].append(emg_block[1][ch] + offset)
            time.sleep(0.005)  # 200 Hz pacing

    threading.Thread(target=collect_data, daemon=True).start()

    # --- Function to update the plot ---
    def update_plot(frame):
        for line, buf in zip(lines, data_buffers):
            line.set_data(x, list(buf))
        return lines

    ani = animation.FuncAnimation(fig, update_plot, interval=33, blit=True)  # ~30 FPS
    print("🔁 Streaming EMG data from all Myos...")
    try:
        plt.show()
    except KeyboardInterrupt:
        print("🛑 Stopping...")

    streamer.stop()
    print("✅ All Myos stopped.")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
