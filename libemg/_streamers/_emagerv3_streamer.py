import platform
import time
import threading
from multiprocessing import Event, Process
from queue import Queue, Empty
from typing import Callable

import numpy as np
import serial  # pyserial
import serial.tools.list_ports

from libemg.shared_memory_manager import SharedMemoryManager


# ============================================================
# Emager3 (v3.0) 8192-byte frames with packed 12-bit EMG + IMU + counter
# ============================================================

class Emager3:
    """
    Reader for Emager v3.0 frame format (8192 bytes total):

      [0]    0xAA
      [1]    0x55
      [2..8065]    EMG payload (8064 bytes) = 5376 packed 12-bit values
      [8066..8173] IMU payload area (108 bytes max)
      [8174]       IMU sample count (expected 8 or 9)
      [8175]       Alignment IMU (ignored)
      [8176..8179] Counter (big-endian uint32)  <-- frame_id
      [8180..8189] Empty (ignored)
      [8190]       0x55
      [8191]       0xAA
    """

    HDR0, HDR1 = 0xAA, 0x55
    TLR0, TLR1 = 0x55, 0xAA

    FRAME_SIZE = 8192

    EMG_START = 2
    EMG_LEN = 8064
    EMG_END = EMG_START + EMG_LEN  # 8066

    IMU_START = 8066
    IMU_AREA_LEN = 108
    IMU_NSAMPLES_I = 8174

    CTR_START = 8176

    TRAILER0_I = 8190
    TRAILER1_I = 8191

    EMG_VALUES_PER_FRAME = 5376
    CHANNELS = 64
    SAMPLES_PER_CH_PER_FRAME = EMG_VALUES_PER_FRAME // CHANNELS  # 84

    IMU_AXES = 6
    IMU_BYTES_PER_SAMPLE = IMU_AXES * 2  # int16 per axis

    def __init__(self, baud_rate: int, com_name=None, vid_pid=(12259, 256)):
        self.com_name = com_name
        self.vid_pid = vid_pid

        ports = list(serial.tools.list_ports.comports())
        com_port = None

        for p in ports:
            if self.com_name is None:
                if (p.vid, p.pid) == self.vid_pid:
                    com_port = p.name if platform.system() == "Windows" else p.device.replace("cu", "tty")
                    break
            else:
                if self.com_name in (p.description or ""):
                    com_port = p.name if platform.system() == "Windows" else p.device.replace("cu", "tty")
                    break

        if com_port is None:
            ports_info = []
            for p in ports:
                dev = getattr(p, "device", None) or getattr(p, "name", None) or "<unknown>"
                desc = getattr(p, "description", "") or "<no description>"
                vid = getattr(p, "vid", None)
                pid = getattr(p, "pid", None)
                ports_info.append(f"{dev} - {desc} (VID: {vid}, PID: {pid})")
            avail = "\n".join(f"  - {pi}" for pi in ports_info) if ports_info else "  (no serial ports found)"
            raise RuntimeError(f"Could not find serial port for Emager3. Available ports:\n{avail}")

        # non-blocking; we buffer ourselves
        self.ser = serial.Serial(com_port, baud_rate, timeout=0)
        self.ser.close()

        self._buf = bytearray()
        self.pos = 0

        # stats
        self.frames_ok = 0
        self.bad_tlr = 0
        self.resyncs = 0
        self.last_ctr = None
        self.ctr_miss = 0

        # handlers
        self.frame_handlers = []

        # imu dtype
        self.imu_dtype = np.dtype(">i2") # if self.imu_endianness == "be" else np.dtype("<i2")
        self._hdr = bytes([self.HDR0, self.HDR1])

    def connect(self):
        self.ser.open()

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    def clear_buffer(self):
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

    def add_frame_handler(self, closure: Callable[[int, np.ndarray, np.ndarray], None]):
        """
        closure(frame_id:int, emg_block:(84,64) uint16, imu_block:(N,6) int16)
        """
        self.frame_handlers.append(closure)

    def _emit_frame(self, frame_id: int, emg_block: np.ndarray, imu_block: np.ndarray):
        for h in self.frame_handlers:
            h(int(frame_id), emg_block, imu_block)

    def _unpack_12bit_be(self, packed: bytes, n_values: int) -> np.ndarray:
        """
        Emager3 (v3.0) 8192-byte frames with packed 12-bit EMG + IMU + counter
        Unpacks the 8064-byte EMG payload into 5376 uint16 values.
        """
        
        b = np.frombuffer(packed, dtype=np.uint8).astype(np.uint16)
        out = np.empty((2 * (len(b) // 3),), dtype=np.uint16)
        out[0::2] = (b[0::3] << 4) | (b[1::3] >> 4)
        out[1::2] = ((b[1::3] & 0x0F) << 8) | b[2::3]
        return out[:n_values]

    def get_data(self) -> bool:
        """
        Returns True if at least one full frame was parsed+emitted in this call.
        """
        try:
            n_av = self.ser.in_waiting
        except Exception:
            return False
        if n_av <= 0:
            return False

        data = self.ser.read(n_av)
        if not data:
            return False

        self._buf += data

        emitted_any = False

        while True:
            h = self._buf.find(self._hdr, self.pos)

            if h < 0:
                keep = min(len(self._buf), self.FRAME_SIZE - 1)
                self._buf = self._buf[-keep:] if keep else bytearray()
                self.pos = 0
                return emitted_any

            if len(self._buf) - h < self.FRAME_SIZE:
                if h > 0:
                    self._buf = self._buf[h:]
                    self.pos = 0
                else:
                    self.pos = h
                return emitted_any

            # validate trailer
            t0 = h + self.TRAILER0_I
            t1 = h + self.TRAILER1_I
            if self._buf[t0] == self.TLR0 and self._buf[t1] == self.TLR1:
                self.frames_ok += 1

                # frame counter big-endian uint32
                c0 = self._buf[h + self.CTR_START + 0]
                c1 = self._buf[h + self.CTR_START + 1]
                c2 = self._buf[h + self.CTR_START + 2]
                c3 = self._buf[h + self.CTR_START + 3]
                frame_id = (c0 << 24) | (c1 << 16) | (c2 << 8) | c3

                if self.last_ctr is not None:
                    expected = (self.last_ctr + 1) & 0xFFFFFFFF
                    if frame_id != expected:
                        self.ctr_miss += 1
                self.last_ctr = frame_id

                # IMU sample count 
                imu_nsamp = int(self._buf[h + self.IMU_NSAMPLES_I])
                imu_bytes_used = imu_nsamp * self.IMU_BYTES_PER_SAMPLE

                emg_bytes = bytes(self._buf[h + self.EMG_START: h + self.EMG_END])
                imu_bytes = bytes(self._buf[h + self.IMU_START: h + self.IMU_START + imu_bytes_used])

                # decode EMG -> (84,64) uint16
                emg_vals = self._unpack_12bit_be(emg_bytes, n_values=self.EMG_VALUES_PER_FRAME)
                emg_block = emg_vals.reshape(self.SAMPLES_PER_CH_PER_FRAME, self.CHANNELS)

                # decode IMU -> (N,6) int16
                imu_arr = np.frombuffer(imu_bytes, dtype=self.imu_dtype)
                imu_block = imu_arr.reshape(imu_nsamp, self.IMU_AXES)

                self._emit_frame(frame_id, emg_block, imu_block)
                emitted_any = True

                self.pos = h + self.FRAME_SIZE
                if self.pos > (self.FRAME_SIZE * 2):
                    self._buf = self._buf[self.pos:]
                    self.pos = 0

            else:
                self.bad_tlr += 1
                self.resyncs += 1
                self.pos = h + 1
                if self.pos > (self.FRAME_SIZE * 2):
                    self._buf = self._buf[self.pos:]
                    self.pos = 0


# ============================================================
# Streamer process (fast path: parse -> enqueue ONE tuple; writer thread updates SMM)
# ============================================================

class EmagerV3Streamer(Process):
    def __init__(self, shared_memory_items, emager_kwargs: dict | None = None):
        super().__init__(daemon=True)
        self.shared_memory_items = shared_memory_items
        self._stop_event = Event()
        self.e = None
        self.emager_kwargs = emager_kwargs or {}

        # cache shapes for ring writes
        self._shapes = {item[0]: item[1] for item in shared_memory_items if len(item) >= 2}

        # writer thread plumbing (created in run)
        self._q = None
        self._writer = None

    def run(self):
        # Create shared memory manager IN CHILD PROCESS
        self.smm = SharedMemoryManager()

        # Create shared memory variables
        for item in self.shared_memory_items:
            self.smm.create_variable(*item)

        # Device
        bw = self.emager_kwargs
        baud = int(bw.get("baud_rate", 3000000))
        com_name = bw.get("com_name", None)
        vid_pid = bw.get("vid_pid", (12259, 256))

        self.e = Emager3(baud_rate=baud, com_name=com_name, vid_pid=vid_pid)
        self.e.connect()
        self.e.clear_buffer()

        # Queue carries ONE bundled item per frame to prevent desync
        self._q = Queue(maxsize=200)  # 1 item/frame; bump if you want

        def buffer_write(tag: str, data: np.ndarray) -> None:
            """
            Prepend `data` (N,D) to the shared memory buffer `tag` (H,D),
            keeping buffer size fixed, and increment `{tag}_count` by N.
            """
            if data is None:
                return

            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.ndim != 2:
                return

            count_tag = f"{tag}_count"

            def add_to_buffer(buffer, new=data):
                # prepend new rows
                new_buffer = np.vstack((new[::-1], buffer))
                # keep buffer size fixed
                return new_buffer[:buffer.shape[0], :]

            # write data
            self.smm.modify_variable(tag, add_to_buffer)

            # increment count by number of rows written
            nb_row = data.shape[0]
            self.smm.modify_variable(count_tag, lambda x, r=nb_row: x + r)

        def writer_thread_fn():
            while not self._stop_event.is_set():
                try:
                    frame_id, emg_block, imu_block = self._q.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    # EMG: store as uint16 (your shared memory is now uint16)
                    emg = np.asarray(emg_block, dtype=np.uint16)

                    # IMU: int16
                    imu = np.asarray(imu_block, dtype=np.int16)

                    # Sample ID per EMG row: (frame_id * CHANNELS) + [0..N-1]
                    base = int(frame_id) * Emager3.SAMPLES_PER_CH_PER_FRAME
                    sample_id = (base + np.arange(emg.shape[0], dtype=np.int64)).reshape(-1, 1)

                    # Write all three (still separate locks per tag, but all-or-nothing per frame at queue level)
                    buffer_write("emg", emg)
                    buffer_write("imu", imu)
                    buffer_write("sample_id", sample_id)

                except Exception:
                    pass
                finally:
                    self._q.task_done()

        self._writer = threading.Thread(target=writer_thread_fn, daemon=True)
        self._writer.start()

        # Frame handler: decode is already done in Emager3; this just enqueues ONE item
        self.drop_count = 0
        def on_frame(frame_id, emg_block, imu_block):
            try:
                self._q.put_nowait((int(frame_id), emg_block, imu_block))
            except Exception:
                self.drop_count += 1
                if self.drop_count % 10 == 0:
                    print("DROPPED frames:", self.drop_count, "qsize:", self._q.qsize())

        self.e.add_frame_handler(on_frame)

        # Main streaming loop (avoid busy spin)
        try:
            while not self._stop_event.is_set():
                did = self.e.get_data()
                if not did:
                    time.sleep(0.001)  # 1 ms backoff when no complete frame parsed
        finally:
            self._cleanup()

    def stop(self):
        self._stop_event.set()
        self.join()

    def _cleanup(self):
        try:
            if self.e is not None:
                self.e.close()
        except Exception:
            pass
        try:
            if hasattr(self, "smm") and self.smm is not None:
                self.smm.cleanup()
        except Exception:
            pass
