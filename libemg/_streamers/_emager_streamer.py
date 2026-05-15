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

class EmagerStreamer(Process):
    def __init__(self, shared_memory_items, version: str = "v1.0", emager_kwargs: dict | None = None):
        """
        :param shared_memory_items: list[(name, shape, dtype, lock)]
        :param version: str Emager version: 'v1.0' or 'v1.1'
        :param emager_kwargs: dict passed to Emager. Supported keys:
          baud_rate (int, default 1500000)
        """
        super().__init__(daemon=True)
        self.smm = SharedMemoryManager()
        self.shared_memory_items = shared_memory_items
        self._stop_event = Event()
        self.e = None

        version = version.strip().lower().lstrip('v').replace('_', '.')
        if '.' not in version:
            version += '.0'
        if version not in ['1.0', '1.1']:
            raise ValueError(f"Unsupported Emager version: {version}. Use 'v1.0' or 'v1.1' (for v3, use emagerv3_streamer).")
        self.version = version
        self.emager_kwargs = emager_kwargs or {}

    def run(self):
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

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

