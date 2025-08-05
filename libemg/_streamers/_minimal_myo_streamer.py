import asyncio
import struct
from multiprocessing import Process
from bleak import BleakClient
from bleak import BleakScanner


# UUIDs
CONTROL_CHAR_UUID = "d5060401-a904-deb9-4748-2c7f4a124842"
EMG_CHAR_UUIDS = [
    "d5060105-a904-deb9-4748-2c7f4a124842",
    "d5060205-a904-deb9-4748-2c7f4a124842",
    "d5060305-a904-deb9-4748-2c7f4a124842",
    "d5060405-a904-deb9-4748-2c7f4a124842",
]

# Commands
INITIAL_COMMANDS = [
    bytes([0x09, 0x01, 0x01]),                          # No sleep
    bytes([0x01, 0x03, 0x02, 0x00, 0x00]),              # EMG filtered, no IMU, no classifier
    bytes([0x03, 0x01, 0x03]),                          # Vibrate long
]

END_COMMANDS = [
    bytes([0x01, 0x03, 0x00, 0x00, 0x00]),              # Stop EMG
    bytes([0x09, 0x01, 0x00]),                          # Normal sleep
    bytes([0x03, 0x01, 0x01]),                          # Vibrate short
]

# Pipe messages
CMD_START_STREAM = "START_STREAM"
CMD_STOP = "STOP"
MSG_EMG_SAMPLE = "EMG_SAMPLE"
MSG_READY = "READY"
MSG_STOPPED = "STOPPED"
MSG_ERROR = "ERROR"

import asyncio
import struct
from multiprocessing import Process, Array, Value
from bleak import BleakClient

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

END_COMMANDS = [
    bytes([0x01, 0x03, 0x00, 0x00, 0x00]),
    bytes([0x09, 0x01, 0x00]),
    bytes([0x03, 0x01, 0x01]),
]

from multiprocessing import Process, Array, Value, Event
import numpy as np

class MinimalMyoBLE(Process):
    def __init__(self, mac_address, channels=8, samples=2):
        super().__init__(daemon=True)
        self.mac = mac_address
        self.shared_data = Array('f', channels*samples)
        self.shared_flag = Value('b', False)
        self.signal = Event()

    def run(self):
        asyncio.run(self.start_stream())

    async def start_stream(self):
        try:
            async with BleakClient(self.mac, timeout=20) as client:
                print("[Myo] Connected")
                await asyncio.sleep(3)
                # Send initial commands
                for cmd in INITIAL_COMMANDS:
                    await client.write_gatt_char(CONTROL_CHAR_UUID, cmd)
                    await asyncio.sleep(0.05)
                await asyncio.sleep(2)
                # Start EMG notifications
                for uuid in EMG_CHAR_UUIDS:
                    await client.start_notify(uuid, self._make_handler())

                print("[Myo] Streaming started")
                await asyncio.sleep(2)
                while not self.signal.is_set():
                    await asyncio.sleep(0.05)

                # Stop EMG streaming gracefully
                # for cmd in END_COMMANDS:
                #     await client.write_gatt_char(CONTROL_CHAR_UUID, cmd)
                #     await asyncio.sleep(0.05)

                print("[Myo] Streaming stopped")

        except Exception as e:
            print(f"[Myo ERROR] {e}")

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
            return [data[:8], data[8:]]
        return None


# from libemg._streamers.minimal_myo_ble_shared import MinimalMyoBLEShared

class MinimalMyoBLEStreamer:
    def __init__(self, mac_address):
        self.device = MinimalMyoBLE(mac_address)

    def start_streaming(self):
        self.device.start()

    def stop(self):
        self.device.stop()

    def read(self):
        return self.device.read()
    

async def scan_ble_devices(timeout=20):
    print("Scanning for BLE devices...")
    devices = await BleakScanner.discover(timeout=timeout)
    if not devices:
        print("No BLE devices found.")
    else:
        print("Found BLE devices:")
        for d in devices:
            name = d.name or "Unknown"
            print(f"Name: {name}, Address: {d.address}, RSSI: {d.rssi}, Metadata: {d.metadata}")

# from libemg.streamers import MinimalMyoBLEStreamer

if __name__ == "__main__":
    import time
    from multiprocessing import freeze_support
    freeze_support()

    # Scan and print BLE devices
    # asyncio.run(scan_ble_devices())


    # Replace with your MAC address
    myo = MinimalMyoBLEStreamer("C2:44:51:6A:0C:4B")
    myo.start_streaming()
    
    time.sleep(10) 

    myo2 = MinimalMyoBLEStreamer("FD:5D:41:18:6E:F8")
    myo2.start_streaming()

    print("Waiting for EMG data...")
    for _ in range(200):
        data = myo.read()
        if data:
            print(data)
        time.sleep(0.05)

    myo.stop()
