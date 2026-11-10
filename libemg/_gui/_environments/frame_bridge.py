"""Carries video frames and input between an environment and the window showing it.

An environment draws with pygame in its own process, so its frames have to
reach the GUI somehow. They do it by both sides pointing at the same memory.

The GUI hands DearPyGui a raw texture backed by a shared-memory segment, and
DearPyGui draws from the array it was given rather than from a copy taken when
the texture was made. The environment writes its frame into that same segment.
Nothing is copied on the GUI side and nothing is uploaded per frame; the
environment paints and the window shows it.

Why the pixels are not locked
-----------------------------
One writer, one reader, and a frame that is replaced whole several times a
second. The worst a race can do is show one frame that is half new and half
old, for one sixtieth of a second, which is invisible. Taking a lock on two
megabytes sixty times a second to prevent that would cost more than it saves,
and would let a stalled reader hold up the game. The small control block beside
the pixels *is* guarded, because a torn integer there would be a real fault.

Input goes the other way through the same mechanism. The environment has no
window of its own, so it receives no keyboard or mouse events; the GUI writes
what it saw into a small block and the environment reads it.
"""

import time
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# Control block layout. Small, fixed, and read far more often than written.
_GENERATION = 0     # bumped once per published frame
_WIDTH = 1
_HEIGHT = 2
_RUNNING = 3        # the environment is alive and drawing
_QUIT = 4           # the GUI has asked it to stop
_FRAMES = 5         # diagnostics: frames published
_FPS_MILLI = 6      # measured frames per second, times 1000
_FINISHED = 7       # the environment ended on its own terms
CONTROL_FIELDS = 8

#: Bytes reserved for a failure message. An environment that refuses its
#: settings does so in its own process, where nothing would otherwise be seen;
#: this is how the reason gets back to the window that launched it.
ERROR_BYTES = 2048

# Input block layout: a slot per key we forward, then the pointer.
MAX_KEYS = 32
_MOUSE_X = MAX_KEYS
_MOUSE_Y = MAX_KEYS + 1
_MOUSE_DOWN = MAX_KEYS + 2
INPUT_FIELDS = MAX_KEYS + 3

#: The keys an embedded environment can be driven with. The index into this
#: tuple is the slot in the input block, so the two sides agree without having
#: to ship a pygame key code across.
FORWARDED_KEYS = ("left", "right", "up", "down",
                  "1", "2", "3", "4",
                  "space", "escape", "w", "a", "s", "d")


class FrameBridge:
    """Shared memory holding one video frame, plus control and input.

    Parameters
    ----------
    name: str
        Prefix for the segments. Unique per running environment.
    width, height: int
        Frame size in pixels. Only meaningful when creating.
    create: bool (optional), default=False
        Create the segments rather than attaching to existing ones.

    Examples
    ---------
    >>> bridge = FrameBridge('fitts_1', 800, 600, create=True)
    >>> texture = bridge.pixels()          # hand this to add_raw_texture
    >>> bridge.close()
    """

    def __init__(self, name, width=0, height=0, create=False, lock=None):
        self.name = name
        self.lock = lock or Lock()
        control_size = CONTROL_FIELDS * 8
        input_size = INPUT_FIELDS * 4

        if create:
            if width <= 0 or height <= 0:
                raise ValueError("Creating a bridge needs a frame size.")
            self.width, self.height = int(width), int(height)
            pixel_size = self.width * self.height * 4 * 4  # float32 RGBA
            self._pixels_sm = _fresh(name + "_pixels", pixel_size)
            self._control_sm = _fresh(name + "_control", control_size)
            self._input_sm = _fresh(name + "_input", input_size)
            self._error_sm = _fresh(name + "_error", ERROR_BYTES)
            self._bind()
            self._pixel_view[:] = 0.0
            self._control[:] = 0
            self._control[_WIDTH] = self.width
            self._control[_HEIGHT] = self.height
            self._input[:] = 0
            self._error[:] = 0
        else:
            self._control_sm = SharedMemory(name + "_control")
            self._control = np.ndarray((CONTROL_FIELDS,), dtype=np.int64,
                                       buffer=self._control_sm.buf)
            self.width = int(self._control[_WIDTH])
            self.height = int(self._control[_HEIGHT])
            pixel_size = self.width * self.height * 4 * 4
            self._pixels_sm = SharedMemory(name + "_pixels")
            self._input_sm = SharedMemory(name + "_input")
            self._error_sm = SharedMemory(name + "_error")
            self._bind(attached=True)

        self._last_published = time.perf_counter()

    def _bind(self, attached=False):
        if not attached:
            self._control = np.ndarray((CONTROL_FIELDS,), dtype=np.int64,
                                       buffer=self._control_sm.buf)
        self._pixel_view = np.ndarray((self.height * self.width * 4,),
                                      dtype=np.float32, buffer=self._pixels_sm.buf)
        self._input = np.ndarray((INPUT_FIELDS,), dtype=np.int32,
                                 buffer=self._input_sm.buf)
        self._error = np.ndarray((ERROR_BYTES,), dtype=np.uint8,
                                 buffer=self._error_sm.buf)

    # ------------------------------------------------------------------
    # the GUI side
    # ------------------------------------------------------------------
    def pixels(self):
        """The frame, flat, as DearPyGui's raw textures want it.

        Hand this array straight to ``add_raw_texture``. DearPyGui keeps the
        array rather than copying it, so once the texture exists the
        environment's writes into this same memory are what the window shows.
        """
        return self._pixel_view

    def generation(self):
        """How many frames have been published. Cheap enough to read per frame."""
        return int(self._control[_GENERATION])

    def running(self):
        return bool(self._control[_RUNNING])

    def finished(self):
        """Whether the environment ended on its own, rather than being stopped."""
        return bool(self._control[_FINISHED])

    def fps(self):
        """Frames per second the environment is actually achieving."""
        return self._control[_FPS_MILLI] / 1000.0

    def frames(self):
        return int(self._control[_FRAMES])

    def request_quit(self):
        """Ask the environment to stop at its next frame."""
        with self.lock:
            self._control[_QUIT] = 1

    def send_input(self, keys, mouse=None, mouse_down=False):
        """Forward what the window saw to the environment.

        Parameters
        ----------
        keys: iterable of str
            Names from :data:`FORWARDED_KEYS` that are currently held.
        mouse: tuple or None (optional)
            Pointer position within the frame, in pixels.
        mouse_down: bool (optional)
            Whether a mouse button is held.
        """
        held = {k for k in keys}
        with self.lock:
            for index, name in enumerate(FORWARDED_KEYS):
                self._input[index] = 1 if name in held else 0
            if mouse is not None:
                self._input[_MOUSE_X] = int(mouse[0])
                self._input[_MOUSE_Y] = int(mouse[1])
            self._input[_MOUSE_DOWN] = 1 if mouse_down else 0

    # ------------------------------------------------------------------
    # the environment side
    # ------------------------------------------------------------------
    def publish(self, surface):
        """Write a pygame surface into the shared frame.

        Parameters
        ----------
        surface: pygame.Surface
            What the environment just drew.
        """
        import pygame
        size = surface.get_size()
        if size != (self.width, self.height):
            # The texture was sized before the first frame arrived, from what
            # the environment said it would draw. An environment that sets up a
            # different size cannot be shown, and saying so beats a shape error
            # from deep inside numpy that names neither side.
            raise ValueError(
                f"This environment drew a {size[0]} by {size[1]} frame, but the "
                f"window was prepared for {self.width} by {self.height}. Its "
                "declared frame size has to match the size it sets up.")
        raw = pygame.image.tobytes(surface, "RGBA")
        # One pass, straight into the texture's own memory: read the bytes as
        # uint8, scale into the float view. No intermediate array survives.
        flat = np.frombuffer(raw, dtype=np.uint8)
        np.divide(flat, 255.0, out=self._pixel_view)
        now = time.perf_counter()
        elapsed = now - self._last_published
        self._last_published = now
        with self.lock:
            self._control[_GENERATION] += 1
            self._control[_FRAMES] += 1
            if elapsed > 0:
                # Smoothed, because a per-frame reciprocal jitters too much to
                # read on a status line.
                previous = self._control[_FPS_MILLI]
                instant = 1000.0 / elapsed
                self._control[_FPS_MILLI] = int(0.9 * previous + 0.1 * instant) \
                    if previous else int(instant)

    def mark_running(self, running=True):
        with self.lock:
            self._control[_RUNNING] = 1 if running else 0

    def mark_finished(self):
        """Record that the environment ended on its own terms."""
        with self.lock:
            self._control[_FINISHED] = 1
            self._control[_RUNNING] = 0

    def report_error(self, message):
        """Record why the environment could not run, for the GUI to show.

        An environment validates its own settings and raises when they
        conflict, but it does that in its own process where the traceback goes
        nowhere a user will look. Writing the reason here is what turns a
        window that stayed black into a sentence explaining what to change.
        """
        encoded = str(message).encode("utf-8")[:ERROR_BYTES - 1]
        with self.lock:
            self._error[:] = 0
            self._error[:len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)

    def error(self):
        """The failure message, or an empty string.

        Returns
        ----------
        str
            Why the environment stopped, when it stopped because of a problem.
        """
        raw = bytes(self._error)
        end = raw.find(bytes([0]))
        return raw[:end if end >= 0 else len(raw)].decode("utf-8", "replace")

    def quit_requested(self):
        return bool(self._control[_QUIT])

    def held_keys(self):
        """Key names the GUI reports as held.

        Returns
        ----------
        set
            Names from :data:`FORWARDED_KEYS`.
        """
        return {name for index, name in enumerate(FORWARDED_KEYS)
                if self._input[index]}

    def pointer(self):
        """Pointer position and button state, as ``(x, y, down)``."""
        return (int(self._input[_MOUSE_X]), int(self._input[_MOUSE_Y]),
                bool(self._input[_MOUSE_DOWN]))

    # ------------------------------------------------------------------
    def close(self, unlink=False):
        """Release the segments. ``unlink`` destroys them, so only the owner."""
        for handle in (self._pixels_sm, self._control_sm, self._input_sm,
                       self._error_sm):
            try:
                handle.close()
                if unlink:
                    handle.unlink()
            except Exception:
                pass

    def __getstate__(self):
        # Only the name and the lock cross a process boundary; the child
        # attaches for itself, because a mapped segment is not portable.
        return {"name": self.name, "lock": self.lock}

    def __setstate__(self, state):
        self.__init__(state["name"], create=False, lock=state["lock"])


def _fresh(name, size):
    """Create a segment, discarding one left behind by a previous run."""
    try:
        stale = SharedMemory(name, create=False)
        stale.close()
        stale.unlink()
    except Exception:
        pass
    return SharedMemory(name, create=True, size=size)
