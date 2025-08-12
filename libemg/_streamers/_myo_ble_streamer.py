import asyncio
import struct
import numpy as np
from multiprocessing import Process, Event, Array, Value, Event, freeze_support
from bleak import BleakClient, BleakScanner
# import bleak.backends.winrt.util as winrt_util
from libemg.shared_memory_manager import SharedMemoryManager
import time
from functools import partial
import sys

# if sys.platform.startswith("win"):
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
# os.environ["BLEAK_USE_PYWINRT_BACKEND"] = "1"

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
    bytes([0x03, 0x01, 0x03]),
]

END_COMMANDS = [
    bytes([0x09, 0x01, 0x00]),
    bytes([0x01, 0x03, 0x00, 0x00, 0x00]),
    bytes([0x03, 0x01, 0x01]),
    bytes([0x03, 0x01, 0x01]),
]

class MyoDevice():
    def __init__(self, mac_address, device_id, smm):
        self.mac = mac_address
        self.num_channels = 8
        self.num_samples = 2
        self.signal = Event()
        self.device_id = device_id
        self.smm = smm
        self.connected = False
    
    async def connect(self):
        # conenct
        try:
            self.client = BleakClient(self.mac, timeout=20)
            # await asyncio.sleep(15)
            # Wait until we can actually connect or timeout after 20s
            await asyncio.wait_for(self.client.connect(), timeout=20)
            print(f"[{self.mac}] Connected")
            # await self.client.connect()
            # print(f"[{self.mac}] Connected")
            await asyncio.sleep(2)

                # no sleep, filtered signal, vibrate 
            for cmd in INITIAL_COMMANDS:
                await self.client.write_gatt_char(CONTROL_CHAR_UUID, cmd)
                await asyncio.sleep(0.05)

            await asyncio.sleep(0.5)

                # for the emg uuids, call the _make_handler on data
            self.handlers = []
            for uuid in EMG_CHAR_UUIDS:
                handler = self._make_handler()
                self.handlers.append(handler)  # Save reference!
                await self.client.start_notify(uuid, handler)
                await asyncio.sleep(0.05)


            self.connected = True
        
        except Exception as e:
            print(f"[{self.mac} ERROR] {e}")

    async def disconnect(self):
        for cmd in END_COMMANDS:
            await self.client.write_gatt_char(CONTROL_CHAR_UUID, cmd)
            await asyncio.sleep(0.05)

        await self.client.disconnect()

    def _make_handler(self):
        # todo: self.smm and self.
        def handler(sender, data: bytearray):
            try:
                emg_vector = [struct.unpack("b", bytes([b]))[0] / 128.0 for b in data]
                emg_vector = np.array(emg_vector).reshape((self.num_samples, self.num_channels))

                existing_emg = self.smm.get_variable("emg")
                channel_range = existing_emg[:-1*(self.num_samples), self.device_id*self.num_channels : (self.device_id+1) * self.num_channels]
                channel_range = np.vstack((emg_vector, channel_range))
                existing_emg[:,self.device_id*self.num_channels : (self.device_id+1) * self.num_channels] = channel_range
                self.smm.modify_variable("emg", lambda x: existing_emg)

                device_count = int(existing_emg.shape[1]/self.num_channels)
                if self.device_id == device_count - 1:
                    self.smm.modify_variable("emg_count", lambda x: x + emg_vector.shape[0])
            except Exception as e:
                print(f"Handler error: {e}")

            
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
        return self.connected


class MyoBLE(Process):
    def __init__(self, mac_addresses, shared_memory_items):
        Process.__init__(self, daemon=True)
        self.signal = Event()
        self.devices = []
        self.mac_addresses = mac_addresses
        self.shared_memory_items = shared_memory_items
        self.smm = SharedMemoryManager()

    def run(self):
        asyncio.run(self.start_streaming())

    async def start_streaming(self):
        # winrt_util.init_apartment()
        # setup shared memory
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)
        
        # connect to the device(s)
        for idx, mac_address in enumerate(self.mac_addresses):
            device = await self.connect(mac_address, idx)
            self.devices.append(device)

        try:
            while not self.signal.is_set():
                await asyncio.sleep(0.1)  # Sleep allows loop to process events/notifications

        except Exception as e:
            print(f"Errored within LibEMG-> MyoBLEStreamer: {e}")

        finally:
            await self._cleanup()
            quit()

    async def connect(self, mac, index, max_retries=5, retry_delay=1):
        attempt = 0
        while attempt < max_retries:
            dev = MyoDevice(mac, index, self.smm)

            print(f"⏳ Waiting for Myo {index + 1} to connect (attempt {attempt + 1})...")
            i = 0
            while i < max_retries:
                await dev.connect()
                if dev.is_connected():
                    print(f"✅ Myo {index + 1} connected!")
                    return dev
                await asyncio.sleep(retry_delay)
                print(f"⚠️ Myo {index + 1} failed to connect. Retrying...")
                attempt += 1
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

    async def _cleanup(self):
        for device in self.devices:
            await device.disconnect()
        self.smm.cleanup()


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

