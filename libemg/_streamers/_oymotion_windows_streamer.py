import asyncio
import struct
from asyncio import Queue
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Dict, List

"""
Thanks to @zubaidah93 for providing the source code. 
"""

import numpy as np
from bleak import BleakScanner, BLEDevice, AdvertisementData, BleakClient, BleakGATTCharacteristic

SERVICE_GUID = '0000ffd0-0000-1000-8000-00805f9b34fb'
CMD_NOTIFY_CHAR_UUID = 'f000ffe1-0451-4000-b000-000000000000'
DATA_NOTIFY_CHAR_UUID = 'f000ffe2-0451-4000-b000-000000000000'


@dataclass
class Characteristic:
    uuid: str
    service_uuid: str
    descriptor_uuids: List[str]


class Command(IntEnum):
    GET_PROTOCOL_VERSION = 0x00,
    GET_FEATURE_MAP = 0x01,
    GET_DEVICE_NAME = 0x02,
    GET_MODEL_NUMBER = 0x03,
    GET_SERIAL_NUMBER = 0x04,
    GET_HW_REVISION = 0x05,
    GET_FW_REVISION = 0x06,
    GET_MANUFACTURER_NAME = 0x07,
    GET_BOOTLOADER_VERSION = 0x0A,

    GET_BATTERY_LEVEL = 0x08,
    GET_TEMPERATURE = 0x09,

    POWEROFF = 0x1D,
    SWITCH_TO_OAD = 0x1E,
    SYSTEM_RESET = 0x1F,
    SWITCH_SERVICE = 0x20,

    SET_LOG_LEVEL = 0x21,
    SET_LOG_MODULE = 0x22,
    PRINT_KERNEL_MSG = 0x23,
    MOTOR_CONTROL = 0x24,
    LED_CONTROL_TEST = 0x25,
    PACKAGE_ID_CONTROL = 0x26,
    SEND_TRAINING_PACKAGE = 0x27,

    GET_ACCELERATE_CAP = 0x30,
    SET_ACCELERATE_CONFIG = 0x31,

    GET_GYROSCOPE_CAP = 0x32,
    SET_GYROSCOPE_CONFIG = 0x33,

    GET_MAGNETOMETER_CAP = 0x34,
    SET_MAGNETOMETER_CONFIG = 0x35,

    GET_EULER_ANGLE_CAP = 0x36,
    SET_EULER_ANGLE_CONFIG = 0x37,

    QUATERNION_CAP = 0x38,
    QUATERNION_CONFIG = 0x39,

    GET_ROTATION_MATRIX_CAP = 0x3A,
    SET_ROTATION_MATRIX_CONFIG = 0x3B,

    GET_GESTURE_CAP = 0x3C,
    SET_GESTURE_CONFIG = 0x3D,

    GET_EMG_RAWDATA_CAP = 0x3E,
    SET_EMG_RAWDATA_CONFIG = 0x3F,

    GET_MOUSE_DATA_CAP = 0x40,
    SET_MOUSE_DATA_CONFIG = 0x41,

    GET_JOYSTICK_DATA_CAP = 0x42,
    SET_JOYSTICK_DATA_CONFIG = 0x43,

    GET_DEVICE_STATUS_CAP = 0x44,
    SET_DEVICE_STATUS_CONFIG = 0x45,

    GET_EMG_RAWDATA_CONFIG = 0x46,

    SET_DATA_NOTIF_SWITCH = 0x4F,
    # Partial command packet, format: [CMD_PARTIAL_DATA, packet number in reverse order, packet content]
    MD_PARTIAL_DATA = 0xFF


class DataSubscription(IntEnum):
    # Data Notify All Off
    OFF = 0x00000000,

    # Accelerate On(C.7)
    ACCELERATE = 0x00000001,

    # Gyroscope On(C.8)
    GYROSCOPE = 0x00000002,

    # Magnetometer On(C.9)
    MAGNETOMETER = 0x00000004,

    # Euler Angle On(C.10)
    EULERANGLE = 0x00000008,

    # Quaternion On(C.11)
    QUATERNION = 0x00000010,

    # Rotation Matrix On(C.12)
    ROTATIONMATRIX = 0x00000020,

    # EMG Gesture On(C.13)
    EMG_GESTURE = 0x00000040,

    # EMG Raw Data On(C.14)
    EMG_RAW = 0x00000080,

    # HID Mouse On(C.15)
    HID_MOUSE = 0x00000100,

    # HID Joystick On(C.16)
    HID_JOYSTICK = 0x00000200,

    # Device Status On(C.17)
    DEVICE_STATUS = 0x00000400,

    # Device Log On
    LOG = 0x00000800,

    # Data Notify All On
    ALL = 0xFFFFFFFF


class DataType(IntEnum):
    ACC = 0x01,
    GYO = 0x02,
    MAG = 0x03,
    EULER = 0x04,
    QUAT = 0x05,
    ROTA = 0x06,
    EMG_GEST = 0x07,
    EMG_ADC = 0x08,
    HID_MOUSE = 0x09,
    HID_JOYSTICK = 0x0A,
    DEV_STATUS = 0x0B,
    LOG = 0x0C,

    PARTIAL = 0xFF


class SampleResolution(IntEnum):
    BITS_8 = 8,
    BITS_12 = 12


class SamplingRate(IntEnum):
    HZ_500 = 500,
    HZ_650 = 650,
    HZ_1000 = 1000


@dataclass
class EmgRawDataConfig:
    fs: SamplingRate = SamplingRate.HZ_1000
    channel_mask: int = 0xFF
    batch_len: int = 32
    resolution: SampleResolution = SampleResolution.BITS_8

    def to_bytes(self):
        body = b''
        body += struct.pack('<H', self.fs)
        body += struct.pack('<H', self.channel_mask)
        body += struct.pack('<B', self.batch_len)
        body += struct.pack('<B', self.resolution)
        return body

    @classmethod
    def from_bytes(cls, data: bytes):
        fs, channel_mask, batch_len, resolution = struct.unpack(
            '@HHBB',
            data,
        )
        return cls(fs, channel_mask, batch_len, resolution)


@dataclass
class Request:
    cmd: Command
    has_res: bool
    body: Optional[bytes] = None


class ResponseCode(IntEnum):
    SUCCESS = 0x00,
    NOT_SUPPORT = 0x01,
    BAD_PARAM = 0x02,
    FAILED = 0x03,
    TIMEOUT = 0x04,
    PARTIAL_PACKET = 0xFF


@dataclass
class Response:
    code: ResponseCode
    cmd: Command
    data: bytes


def _match_nus_uuid(_device: BLEDevice, adv: AdvertisementData):
    if SERVICE_GUID.lower() in adv.service_uuids:
        return True
    return False

from multiprocessing import Process, Event
from libemg.shared_memory_manager import SharedMemoryManager
import numpy as np
class Gforce(Process):
    def __init__(self, sampling_rate=1000, res=8, emg=True, imu=False, shared_memory_items=[]):
        Process.__init__(self, daemon=True)
        self.emg = emg
        self.imu = imu
        self.sampling_rate = sampling_rate
        self.res = res
        self.emg_conf = EmgRawDataConfig()
        if self.sampling_rate == 500:
            self.emg_conf = EmgRawDataConfig(fs = SamplingRate.HZ_500, resolution=SampleResolution.BITS_12)
        
        self.shared_memory_items = shared_memory_items
        self.smm = SharedMemoryManager()
        self.signal = Event()

        self.client = None # bluetooth client
        self.cmd_char = None
        self.data_char = None
        self.responses: Dict[Command, Queue] = {}
        self.resolution = SampleResolution.BITS_12

        self.packet_id = 0
        self.data_packet = []

    async def connect(self):
        device = await BleakScanner.find_device_by_filter(_match_nus_uuid)
        if device is None:
            raise Exception("No GForce device found")

        def handle_disconnect(_: BleakClient):
            for task in asyncio.all_tasks():
                task.cancel()

        client = BleakClient(device, disconnected_callback=handle_disconnect)
        await client.connect()

        await client.start_notify(
            CMD_NOTIFY_CHAR_UUID, self._on_cmd_response,
        )

        self.client = client

    def _on_data_response(self, q: Queue, bs: bytearray):
        bs = bytes(bs)
        full_packet = []

        is_partial_data = bs[0] == ResponseCode.PARTIAL_PACKET
        if is_partial_data:
            packet_id = bs[1]
            if self.packet_id != 0 and self.packet_id != packet_id + 1:
                raise Exception("Unexpected packet id: expected {} got {}".format(
                    self.packet_id + 1,
                    packet_id,
                ))
            elif self.packet_id == 0 or self.packet_id > packet_id:
                self.packet_id = packet_id
                self.data_packet += bs[2:]

                if self.packet_id == 0:
                    full_packet = self.data_packet
                    self.data_packet = []
        else:
            full_packet = bs

        if len(full_packet) == 0:
            return

        data = None
        data_type = DataType(full_packet[0])
        packet = full_packet[1:]
        if data_type == DataType.EMG_ADC:
            data = self._convert_emg_to_uv(packet)
        elif data_type == DataType.ACC:
            data = self._convert_acceleration_to_g(packet)
        elif data_type == DataType.GYO:
            data = self._convert_gyro_to_dps(packet)
        elif data_type == DataType.MAG:
            data = self._convert_magnetometer_to_ut(packet)
        elif data_type == DataType.EULER:
            data = self._convert_euler(packet)
        elif data_type == DataType.QUAT:
            data = self._convert_quaternion(packet)
        elif data_type == DataType.ROTA:
            data = self._convert_rotation_matrix(packet)
        else:
            raise Exception(f"Unknown data type {data_type}, full packet: {full_packet}")

        q.put_nowait(data)

    def _convert_emg_to_uv(self, data: bytes):
        min_voltage = -1.25
        max_voltage = 1.25

        if self.resolution == SampleResolution.BITS_8:
            dtype = np.uint8
            div = 127.0
            sub = 128
        elif self.resolution == SampleResolution.BITS_12:
            dtype = np.uint16
            div = 2047.0
            sub = 2048
        else:
            raise Exception(f"Unsupported resolution {self.resolution}")

        gain = 1200.0
        conversion_factor = (max_voltage - min_voltage) / gain / div

        emg_data = (np.frombuffer(data, dtype=dtype).astype(np.float32) - sub) * conversion_factor
        num_channels = 8

        return emg_data.reshape(-1, num_channels)

    @staticmethod
    def _convert_acceleration_to_g(data: bytes):
        normalizing_factor = 65536.0

        acceleration_data = np.frombuffer(data, dtype=np.int32).astype(np.float32) / normalizing_factor
        num_channels = 3

        return acceleration_data.reshape(-1, num_channels)

    @staticmethod
    def _convert_gyro_to_dps(data: bytes):
        normalizing_factor = 65536.0

        gyro_data = np.frombuffer(data, dtype=np.int32).astype(np.float32) / normalizing_factor
        num_channels = 3

        return gyro_data.reshape(-1, num_channels)

    @staticmethod
    def _convert_magnetometer_to_ut(data: bytes):
        normalizing_factor = 65536.0

        magnetometer_data = np.frombuffer(data, dtype=np.int32).astype(np.float32) / normalizing_factor
        num_channels = 3

        return magnetometer_data.reshape(-1, num_channels)

    @staticmethod
    def _convert_euler(data: bytes):

        euler_data = np.frombuffer(data, dtype=np.float32).astype(np.float32)
        num_channels = 3

        return euler_data.reshape(-1, num_channels)

    @staticmethod
    def _convert_quaternion(data: bytes):

        quaternion_data = np.frombuffer(data, dtype=np.float32).astype(np.float32)
        num_channels = 4

        return quaternion_data.reshape(-1, num_channels)

    @staticmethod
    def _convert_rotation_matrix(data: bytes):

        rotation_matrix_data = np.frombuffer(data, dtype=np.int32).astype(np.float32)
        num_channels = 9

        return rotation_matrix_data.reshape(-1, num_channels)

    @staticmethod
    def _convert_emg_gesture(data: bytes):

        emg_gesture_data = np.frombuffer(data, dtype=np.int16).astype(np.float16)
        num_channels = 6

        return emg_gesture_data.reshape(-1, num_channels)

    def _on_cmd_response(self, _: BleakGATTCharacteristic, bs: bytearray):
        try:
            response = self._parse_response(bytes(bs))
            if response.cmd in self.responses:
                self.responses[response.cmd].put_nowait(
                    response.data,
                )
        except Exception as e:
            raise Exception("Failed to parse response: %s" % e)

    @staticmethod
    def _parse_response(res: bytes):
        code = int.from_bytes(res[:1], byteorder='big')
        code = ResponseCode(code)

        cmd = int.from_bytes(res[1:2], byteorder='big')
        cmd = Command(cmd)

        data = res[2:]

        return Response(
            code=code,
            cmd=cmd,
            data=data,
        )

    async def get_protocol_version(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_PROTOCOL_VERSION,
            has_res=True,
        ))
        return buf.decode('utf-8')

    async def get_feature_map(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_FEATURE_MAP,
            has_res=True,
        ))
        return int.from_bytes(buf, byteorder='big')  # TODO: check if this is correct

    async def get_device_name(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_DEVICE_NAME,
            has_res=True,
        ))
        return buf.decode('utf-8')

    async def get_firmware_revision(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_FW_REVISION,
            has_res=True,
        ))
        return buf.decode('utf-8')

    async def get_hardware_revision(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_HW_REVISION,
            has_res=True,
        ))
        return buf.decode('utf-8')

    async def get_model_number(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_MODEL_NUMBER,
            has_res=True,
        ))
        return buf.decode('utf-8')

    async def get_serial_number(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_SERIAL_NUMBER,
            has_res=True,
        ))
        return buf.decode('utf-8')

    async def get_manufacturer_name(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_MANUFACTURER_NAME,
            has_res=True,
        ))

        return buf.decode('utf-8')

    async def get_bootloader_version(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_BOOTLOADER_VERSION,
            has_res=True,
        ))

        return buf.decode('utf-8')

    async def get_battery_level(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_BATTERY_LEVEL,
            has_res=True,
        ))
        return int.from_bytes(buf, byteorder='big')

    async def get_temperature(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_TEMPERATURE,
            has_res=True,
        ))
        return int.from_bytes(buf, byteorder='big')

    async def power_off(self):
        await self._send_request(Request(
            cmd=Command.POWEROFF,
            has_res=False,
        ))

    async def switch_to_oad(self):
        await self._send_request(Request(
            cmd=Command.SWITCH_TO_OAD,
            has_res=False,
        ))

    async def system_reset(self):
        await self._send_request(Request(
            cmd=Command.SYSTEM_RESET,
            has_res=False,
        ))

    async def switch_service(self):
        await self._send_request(Request(
            cmd=Command.SWITCH_SERVICE,
            has_res=False,
        ))

    async def set_motor(self):  # TODO: check if this works and what it does
        await self._send_request(Request(
            cmd=Command.MOTOR_CONTROL,
            has_res=True,
        ))

    async def set_led(self):  # TODO: check if this works and what it does
        await self._send_request(Request(
            cmd=Command.LED_CONTROL_TEST,
            has_res=True,
        ))

    async def set_log_level(self):
        await self._send_request(Request(
            cmd=Command.SET_LOG_LEVEL,
            has_res=False,
        ))

    async def set_log_module(self):
        await self._send_request(Request(
            cmd=Command.SET_LOG_MODULE,
            has_res=False,
        ))

    async def print_kernel_msg(self):
        await self._send_request(Request(
            cmd=Command.PRINT_KERNEL_MSG,
            has_res=True,
        ))

    async def set_package_id(self):
        await self._send_request(Request(
            cmd=Command.PACKAGE_ID_CONTROL,
            has_res=False,
        ))

    async def send_training_package(self):
        await self._send_request(Request(
            cmd=Command.SEND_TRAINING_PACKAGE,
            has_res=False,
        ))

    async def set_emg_raw_data_config(self, cfg=EmgRawDataConfig()):
        body = cfg.to_bytes()
        await self._send_request(Request(
            cmd=Command.SET_EMG_RAWDATA_CONFIG,
            body=body,
            has_res=True,
        ))
        self.resolution = cfg.resolution

    async def get_emg_raw_data_config(self):
        buf = await self._send_request(Request(
            cmd=Command.GET_EMG_RAWDATA_CONFIG,
            has_res=True,
        ))
        return EmgRawDataConfig.from_bytes(buf)

    async def set_subscription(self, subscription: DataSubscription):
        body = [0xFF & subscription, 0xFF & (subscription >> 8), 0xFF & (subscription >> 16),
                0xFF & (subscription >> 24)]
        body = bytes(body)
        await self._send_request(Request(
            cmd=Command.SET_DATA_NOTIF_SWITCH,
            body=body,
            has_res=True,
        ))

    async def start_streaming(self):
        q = Queue()
        await self.client.start_notify(
            DATA_NOTIFY_CHAR_UUID,
            lambda _, data: self._on_data_response(q, data),
        )
        return q

    async def stop_streaming(self):
        exceptions = []
        try:
            await self.set_subscription(DataSubscription.OFF)
        except Exception as e:
            exceptions.append(e)
        try:
            await self.client.stop_notify(DATA_NOTIFY_CHAR_UUID)
        except Exception as e:
            exceptions.append(e)
        try:
            await self.client.stop_notify(CMD_NOTIFY_CHAR_UUID)
        except Exception as e:
            exceptions.append(e)

        if len(exceptions) > 0:
            raise Exception("Failed to stop streaming: %s" % exceptions)

    async def disconnect(self):
        with suppress(asyncio.CancelledError):
            await self.client.disconnect()

    def _get_response_channel(self, cmd: Command):
        q = Queue()
        self.responses[cmd] = q
        return q

    async def _send_request(self, req: Request):
        q = None
        if req.has_res:
            q = self._get_response_channel(req.cmd)

        bs = bytes([req.cmd])
        if req.body is not None:
            bs += req.body
        await self.client.write_gatt_char(CMD_NOTIFY_CHAR_UUID, bs)

        if not req.has_res:
            return None

        return await asyncio.wait_for(q.get(), 5)

    def run(self):
        asyncio.run(self.start_stream())

    async def start_stream(self):
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)
        await self.connect()
        await self.set_emg_raw_data_config(self.emg_conf)
        await self.set_subscription(
            DataSubscription.EMG_RAW
        )
        print("Connected to Oymotion Cuff!")

        q = await self.start_streaming()
        while True:
            if self.signal.is_set():
                self.cleanup()
                break
            try:
                for e in await q.get():
                    emg = np.expand_dims(np.array(e),0)
                    self.smm.modify_variable("emg", lambda x: np.vstack((emg, x))[:x.shape[0],:])
                    self.smm.modify_variable("emg_count", lambda x: x + emg.shape[0])
                    
            except:
                print("Worker Stopped.")
                quit()

    def cleanup(self):
        self.disconnect()
        print("Oymotion has disconnected.")