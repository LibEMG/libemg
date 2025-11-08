import serial # pyserial
import numpy as np
import platform
from multiprocessing import Event, Process
from queue import Queue, Empty
import threading
from libemg.shared_memory_manager import SharedMemoryManager


def _get_channel_map(version: str = "1.0"):
    if version == "1.1":
        channel_map = [44, 49, 43, 55, 39, 59, 33, 2, 32, 3, 26, 6, 22, 13, 16, 10] + \
                        [42, 48, 45, 54, 38, 58, 35, 0, 34, 1, 27, 7, 23, 11, 17, 12] + \
                        [46, 52, 40, 51, 36, 56, 31, 60, 30, 63, 25, 4, 21, 8, 18, 15] + \
                        [47, 50, 41, 53, 37, 57, 29, 62, 28, 61, 24, 5, 19, 9, 20, 14]      
    else:
        channel_map = [10, 22, 12, 24, 13, 26, 7, 28, 1, 30, 59, 32, 53, 34, 48, 36] + \
                        [62, 16, 14, 21, 11, 27, 5, 33, 63, 39, 57, 45, 51, 44, 50, 40] + \
                        [8, 18, 15, 19, 9, 25, 3, 31, 61, 37, 55, 43, 49, 46, 52, 38] + \
                        [6, 20, 4, 17, 2, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42]

    return channel_map


def reorder(data, mask, match_result):
    '''
    Looks for mask/template matching in data array and reorders
    :param data: (numpy array) - 1D data input
    :param mask: (numpy array) - 1D mask to be matched
    :param match_result: (int) - Expected result of mask-data convolution matching
    :return: (numpy array) - Reordered data array
    '''
    number_of_packet = int(len(data)/128)
    roll_data = []
    for i in range(number_of_packet):
        data_lsb = data[i*128:(i+1)*128] & np.ones(128, dtype=np.int8)
        mask_match = np.convolve(mask, np.append(data_lsb, data_lsb), 'valid')
        try:
            offset = np.where(mask_match == match_result)[0][0] - 3
        except IndexError:
            return None
        roll_data.append(np.roll(data[i*128:(i+1)*128], -offset))
    return roll_data


class Emager:
    def __init__(self, baud_rate, version: str = "1.0"):
        com_name = 'KitProg3'
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if com_name in p.description:
                if platform.system() == 'Windows':
                    com_port = p.name
                else:
                    # Different port names for Mac / Linux (has been tested on Mac but not Linux)
                    com_port = p.device.replace('cu', 'tty')    # using the 'cu' port on Mac doesn't work, so renaming it to 'tty' port
        self.ser = serial.Serial(com_port,baud_rate, timeout=1)
        self.ser.close()

        self.bytes_to_read = 128
        ### ^ Number of bytes in message (i.e. channel bytes + header/tail bytes)
        self.mask = np.array([0, 2] + [0, 1] * 63)
        ### ^ Template mask for template matching on input data
        self.channel_map = _get_channel_map(version)
        self.emg_handlers = []

    def connect(self):
        self.ser.open()
        return

    def add_emg_handler(self, closure):
        self.emg_handlers.append(closure)

    def get_data(self):
        # get and organize data
        bytes_available = self.ser.inWaiting()
        bytesToRead = bytes_available - (bytes_available % 128)
        data_packet = reorder(list(self.ser.read(bytesToRead)), self.mask, 63)
        if data_packet is None or len(data_packet) == 0:
            # No data
            return
        for p in range(len(data_packet)):
            samples = [int.from_bytes(bytes([data_packet[p][s*2], data_packet[p][s*2+1]]), 'big',signed=True) for s in range(64)]
            samples = np.array(samples)[self.channel_map]    # sort columns so columns correspond to channels in ascending order
            for h in self.emg_handlers:
                h(samples)
    
    def clear_buffer(self):
        '''
        Clear the serial port input buffer.
        :return: None
        '''
        self.ser.reset_input_buffer()
        return

    def close(self):
        self.ser.close()
        return

class Emager3:
    """Reader for the new Emager3 device which sends framed payloads:
    HDR 0xAA55, APP_PAYLOAD=8192 bytes (64x64 samples x 2 bytes), TLR 0x55AA.
    The payload is interpreted as 4096 16-bit samples arranged as (time x channel)
    when reshaped to (64,64) and transposed to (channel x time). We emit one
    64-channel sample vector to handlers per timepoint (64 vectors per frame).
    """
    HDR = b"\xAA\x55"
    TLR = b"\x55\xAA"

    def __init__(self, baud_rate, endianness='le', signed=False, com_name=None, vid_pid=(12259, 256),
                 channels: int = 64, samples_per_frame: int = 64, version: str = "3.0"):
        self.com_name = com_name
        self.vid_pid = vid_pid
        ports = list(serial.tools.list_ports.comports())
        com_port = None
        for p in ports:
            if self.com_name is None:
                if (p.vid, p.pid) == self.vid_pid:
                    if platform.system() == 'Windows':
                        com_port = p.name
                    else:
                        com_port = p.device.replace('cu', 'tty')
                    break
            else:
                if self.com_name in p.description:
                    if platform.system() == 'Windows':
                        com_port = p.name
                    else:
                        com_port = p.device.replace('cu', 'tty')
                    break

        if com_port is None:
            print(f"Could not find serial port for {self.com_name}")
            # include a helpful error that lists all detected serial ports
            ports_info = []
            for p in ports:
                dev = getattr(p, "device", None) or getattr(p, "name", None) or "<unknown>"
                desc = getattr(p, "description", "") or "<no description>"
                vid = getattr(p, "vid", None)
                pid = getattr(p, "pid", None)
                ports_info.append(f"{dev} - {desc} (VID: {vid}, PID: {pid})")

            if ports_info:
                avail = "\n".join(f"  - {pi}" for pi in ports_info)
            else:
                avail = "  (no serial ports found)"

            raise RuntimeError(f"Could not find serial port for {self.com_name}. Available ports:\n{avail}")

        self.ser = serial.Serial(com_port, baud_rate, timeout=1)
        self.ser.close()
        self._buf = bytearray()
        self.emg_handlers = []

        # dtype selection
        if endianness == 'le':
            self.sample_dtype = np.int16 if signed else np.uint16
        else:
            self.sample_dtype = np.dtype('>i2') if signed else np.dtype('>u2')

        # framing params
        self.channels = int(channels)
        self.samples_per_frame = int(samples_per_frame)
        self.expected_samples = self.channels * self.samples_per_frame
        self.APP_PAYLOAD = self.expected_samples * 2
        self.FRAME_SIZE = 2 + self.APP_PAYLOAD + 2
        # frame and counter stats
        self.frames_ok = 0
        self.bad_tlr = 0
        self.resyncs = 0
        self.last_ctr = None
        self.ctr_miss = 0
        # parser position for incremental search
        self.pos = 0

    def connect(self):
        self.ser.open()

    def add_emg_handler(self, closure):
        self.emg_handlers.append(closure)

    def clear_buffer(self):
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def _process_frame_payload(self, payload_bytes):
        # Decode payload into a full block (samples_per_frame x channels) and emit once per frame.
        try:
            arr = np.frombuffer(payload_bytes, dtype=self.sample_dtype)
            if arr.size != self.expected_samples:
                return
            # payload is time-major: samples_per_frame x channels
            block_time_ch = arr.reshape(self.samples_per_frame, self.channels)

            # Emit the whole block once to each handler (shape: samples x channels)
            for h in self.emg_handlers:
                try:
                    h(block_time_ch)
                except Exception:
                    pass
        except Exception:
            return

    def get_data(self):
        # Read what's available and append to buffer, then parse frames
        try:
            n_av = self.ser.in_waiting
        except Exception:
            return

        if n_av <= 0:
            return

        data = self.ser.read(n_av)
        if not data:
            return

        self._buf += data

        # incremental frame parsing (mirrors FrameParser.feed logic)
        while True:
            h = self._buf.find(self.HDR, self.pos)
            if h < 0:
                keep = min(len(self._buf), self.FRAME_SIZE - 1)
                if keep:
                    self._buf[:] = self._buf[-keep:]
                else:
                    self._buf.clear()
                self.pos = 0
                return

            if len(self._buf) - h < self.FRAME_SIZE:
                if h > 0:
                    self._buf[:] = self._buf[h:]
                    self.pos = 0
                else:
                    self.pos = h
                return

            t0 = h + 2 + self.APP_PAYLOAD
            if self._buf[t0:t0+2] == self.TLR:
                # good frame
                self.frames_ok += 1
                p = h + 2
                # counter (first 4 bytes of payload, big-endian)
                try:
                    ctr = ((self._buf[p] << 24) |
                           (self._buf[p+1] << 16) |
                           (self._buf[p+2] << 8) |
                            self._buf[p+3])
                    if self.last_ctr is not None:
                        expected = (self.last_ctr + 1) & 0xFFFFFFFF
                        if ctr != expected:
                            self.ctr_miss += 1
                    self.last_ctr = ctr
                except Exception:
                    pass

                try:
                    payload_bytes = bytes(self._buf[p : p + self.APP_PAYLOAD])
                    self._process_frame_payload(payload_bytes)
                except Exception:
                    pass

                self.pos = h + self.FRAME_SIZE
                if self.pos > (self.FRAME_SIZE * 2):
                    self._buf[:] = self._buf[self.pos:]
                    self.pos = 0
            else:
                self.bad_tlr += 1
                self.resyncs += 1
                self.pos = h + 1
                if self.pos > (self.FRAME_SIZE * 2):
                    self._buf[:] = self._buf[self.pos:]
                    self.pos = 0

class EmagerStreamer(Process):
    def __init__(self, shared_memory_items, version: str = "v1", emager_kwargs: dict | None = None):
        """
        :param shared_memory_items: list[(name, shape, dtype, lock)]
        :param version: str Emager version: 'v1.0', 'v1.1', 'v3.0'
        :param emager_kwargs: dict passed to Emager/Emager3. Supported keys:
          baud_rate (int, default 1500000 or 5000000 for v3), endianness ('le'), signed (bool),
          com_name, vid_pid (tuple), channels (int), samples_per_frame (int)
        """
        super().__init__(daemon=True)
        self.smm = SharedMemoryManager()
        self.shared_memory_items = shared_memory_items
        self._stop_event = Event()
        self.e = None
        
        version = version.strip().lower().lstrip('v').replace('_', '.')
        if '.' not in version:
            version += '.0'
        if version not in ['1.0', '1.1', '3.0']:
            raise ValueError(f"Unsupported Emager version: {version}")
        self.version = version
        self.emager_kwargs = emager_kwargs or {}

    def run(self):
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)
        
        # Instantiate the appropriate Emager reader based on version and kwargs
        if self.version == "3.0":
            bw = self.emager_kwargs
            baud = bw.get('baud_rate', 5000000)
            endianness = bw.get('endianness', 'le')
            signed = bw.get('signed', False)
            com_name = bw.get('com_name', None)
            vid_pid = bw.get('vid_pid', (12259, 256))
            channels = bw.get('channels', 64)
            samples_per_frame = bw.get('samples_per_frame', 64)
            self.e = Emager3(baud, endianness=endianness, signed=signed, com_name=com_name, vid_pid=vid_pid,
                              channels=channels, samples_per_frame=samples_per_frame, version=self.version)
        else:
            baud = self.emager_kwargs.get('baud_rate', 1500000)
            self.e = Emager(baud, version=self.version)
        self.e.connect()
        # Create a queue and writer thread to offload shared-memory writes
        q: Queue = Queue(maxsize=100)

        def writer_thread_fn():
            while not self._stop_event.is_set():
                try:
                    block = q.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    # block is samples x channels; stack new rows on top and keep window
                    self.smm.modify_variable('emg', lambda x, b=block: np.vstack((b, x))[:x.shape[0], :])
                    # increment count by number of rows written
                    rows = block.shape[0] if hasattr(block, 'shape') else 1
                    self.smm.modify_variable('emg_count', lambda x, r=rows: x + r)
                except Exception:
                    pass
                finally:
                    q.task_done()

        writer = threading.Thread(target=writer_thread_fn, daemon=True)
        writer.start()

        def write_emg(emg_block):
            # emg_block expected shape: samples x channels (numpy array)
            try:
                q.put_nowait(np.array(emg_block))
            except Exception:
                # if queue full, drop
                pass

        self.e.add_emg_handler(write_emg)

        try:
            if self.e.ser.closed == True:
                self.e.ser.open()
            self.e.clear_buffer()
            while not self._stop_event.is_set():
                self.e.get_data()
        finally:
            self._cleanup()

    def stop(self):
        self._stop_event.set()
        self.join()

    def _cleanup(self):
        if self.e is not None:
            self.e.close()
        self.smm.cleanup()

