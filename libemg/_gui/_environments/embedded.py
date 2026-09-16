"""Run a LibEMG environment offscreen and publish its frames to the GUI.

The environments are pygame games with their own window and their own loop.
Embedding one means taking away the window without taking away the game.

SDL's dummy video driver does exactly that: ``pygame.display.set_mode`` still
returns a real surface and everything still draws onto it, there is simply no
window on screen. The environment is otherwise untouched. Its
:meth:`game_setup` and :meth:`_run_loop` run exactly as they always did, so an
embedded game behaves the same as a windowed one and a new environment needs no
special support to be embeddable.

What has to be replaced is the two things a window used to provide. Frames go
out through a :class:`~libemg._gui._environments.frame_bridge.FrameBridge` into
the texture the GUI is drawing. Input comes back the same way, because with no
window there are no keyboard events, and a keyboard-driven environment asks
pygame for key state rather than reading the queue.
"""

import os
import traceback
from multiprocessing import Process

from libemg._gui._environments.frame_bridge import FORWARDED_KEYS, FrameBridge


class EnvironmentRunner(Process):
    """Runs one environment offscreen, publishing every frame it draws.

    Parameters
    ----------
    bridge: FrameBridge
        Where frames go and input comes from.
    factory: callable
        Called in this process to build the environment. It must be picklable,
        so it should be a module-level function or a small class holding
        configuration rather than a constructed environment: a pygame object
        cannot cross a process boundary, and neither can a socket.
    forward_keys: bool (optional), default=True
        Make ``pygame.key.get_pressed`` reflect the keys the GUI forwards.
        Needed by keyboard-driven environments; harmless otherwise.
    """

    def __init__(self, bridge, factory, forward_keys=True):
        super().__init__(daemon=True)
        self.bridge = bridge
        self.factory = factory
        self.forward_keys = forward_keys

    def run(self):
        # Set before pygame is imported anywhere in this process: the driver is
        # chosen when the display module initialises, and the dummy driver is
        # what makes set_mode produce a surface with no window.
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
        import pygame

        bridge = self.bridge
        try:
            pygame.init()
            pygame.font.init()
            try:
                pygame.mixer.init()
            except Exception:
                # There is no audio device on a headless machine, and a game
                # that cannot play a sound should still be playable.
                pass

            if self.forward_keys:
                _forward_key_state(pygame, bridge)

            environment = self.factory()
            environment.game_setup()
            surface = pygame.display.get_surface()
            clock = pygame.time.Clock()
            fps = int(getattr(environment, "fps", 60) or 60)
            bridge.mark_running(True)

            while not environment.done:
                if bridge.quit_requested():
                    break
                _pump(pygame, bridge)
                environment._run_loop()
                pygame.display.update()
                # get_surface again each frame, because an environment is free
                # to call set_mode itself and replace it.
                surface = pygame.display.get_surface() or surface
                if surface is not None:
                    bridge.publish(surface)
                clock.tick(fps)

            if environment.done:
                bridge.mark_finished()
            try:
                environment.save_results()
            except Exception:
                pass
        except Exception as error:
            # The reason goes back through the bridge as well as to the
            # console. An environment validates its own settings and raises
            # here, in a process with no window and no terminal anybody is
            # watching, so without this the window that launched it simply
            # stays black with nothing to explain why.
            bridge.report_error(f"{type(error).__name__}: {error}")
            print("LibEMG -> embedded environment failed:\n"
                  + traceback.format_exc())
        finally:
            bridge.mark_running(False)
            try:
                import pygame as _pygame
                _pygame.quit()
            except Exception:
                pass
            bridge.close()


def _pump(pygame, bridge):
    """Keep SDL's own queue drained and deliver a quit as a real event.

    Environments read ``pygame.event.get()`` and look for QUIT, so asking one
    to stop is best done in the language it already speaks.
    """
    pygame.event.pump()
    if bridge.quit_requested():
        pygame.event.post(pygame.event.Event(pygame.QUIT))


def _forward_key_state(pygame, bridge):
    """Make pygame report the keys the GUI is forwarding.

    A keyboard-driven environment calls ``pygame.key.get_pressed`` and indexes
    the result by key code. With no window, SDL has no key state to report, so
    that call is replaced by one that reads what the GUI wrote into the bridge.

    Replacing the function rather than posting synthetic events is deliberate.
    ``get_pressed`` reflects SDL's own view of the physical keyboard, which
    posted events do not reach, so posting them would look right and do
    nothing. The replacement lives only in this process, so the library behaves
    exactly as before anywhere else.
    """
    codes = {}
    for name in FORWARDED_KEYS:
        # pygame names the letter and digit keys in lower case (K_a, K_1) and
        # the named ones in upper (K_LEFT, K_SPACE), so both spellings are
        # tried. Getting this wrong silently drops the arrow keys, which are
        # the ones a Fitts task is actually driven with.
        code = getattr(pygame, f"K_{name}", None)
        if code is None:
            code = getattr(pygame, f"K_{name.upper()}", None)
        if code is not None:
            codes[name] = code

    def get_pressed():
        return _KeyState({codes[name] for name in bridge.held_keys()
                          if name in codes})

    pygame.key.get_pressed = get_pressed


class _KeyState:
    """Key state indexed by key code, for any code that is asked for.

    The real ``get_pressed`` returns a sequence covering every scancode SDL
    knows, and callers index it with whatever key code they care about. A list
    sized to the keys being forwarded looks equivalent and is not: indexing it
    with an arrow key, whose code is in the millions, raises IndexError instead
    of answering "not held".
    """

    __slots__ = ("_held",)

    def __init__(self, held):
        self._held = held

    def __getitem__(self, code):
        return code in self._held

    def __contains__(self, code):
        return code in self._held

    def __len__(self):
        # Large enough that a caller checking bounds is satisfied, and never
        # actually iterated over in practice.
        return 1 << 31

    def __iter__(self):
        raise TypeError("Key state is meant to be indexed by key code, not iterated.")


class EmbeddedEnvironment:
    """An environment running offscreen, and the frame the GUI should draw.

    Owns the bridge and the process, so stopping this stops both and releases
    the shared memory.

    Parameters
    ----------
    name: str
        Unique name for this run, used for the shared segments.
    factory: callable
        Builds the environment, in the child process. Must be picklable.
    width, height: int
        Frame size, which must match what the environment sets up.
    forward_keys: bool (optional), default=True
        Whether to make forwarded keys visible to the environment.

    Examples
    ---------
    >>> embedded = EmbeddedEnvironment('fitts_1', factory, 800, 600).start()
    >>> texture = embedded.pixels()
    >>> embedded.stop()
    """

    def __init__(self, name, factory, width, height, forward_keys=True):
        self.name = name
        self.factory = factory
        self.width, self.height = int(width), int(height)
        self.bridge = FrameBridge(name, self.width, self.height, create=True)
        self.runner = EnvironmentRunner(self.bridge, factory,
                                        forward_keys=forward_keys)
        self._started = False

    def start(self):
        if not self._started:
            self.runner.start()
            self._started = True
        return self

    @property
    def alive(self):
        return self._started and self.runner.is_alive()

    def pixels(self):
        """The frame array to back a raw texture with."""
        return self.bridge.pixels()

    def status(self):
        """What the environment is doing.

        Returns
        ----------
        dict
            ``running``, ``finished``, ``frames``, ``fps``, ``generation`` and
            ``error``, the last being why it stopped when it stopped badly.
        """
        return {"running": self.bridge.running(),
                "finished": self.bridge.finished(),
                "frames": self.bridge.frames(),
                "fps": self.bridge.fps(),
                "generation": self.bridge.generation(),
                "error": self.bridge.error()}

    def send_input(self, keys, mouse=None, mouse_down=False):
        self.bridge.send_input(keys, mouse=mouse, mouse_down=mouse_down)

    def stop(self, timeout=4.0):
        """Ask it to stop, wait, and release everything."""
        if self._started:
            self.bridge.request_quit()
            self.runner.join(timeout=timeout)
            if self.runner.is_alive():
                self.runner.terminate()
                self.runner.join(timeout=1.0)
            self._started = False
        self.bridge.close(unlink=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False
