"""Coverage for disposing of interactive plots without ambushing worker threads.

A matplotlib window built on Tk leaves around twenty tkinter objects behind when
it closes. Finalizing one of those from a thread that is not running the Tk main
loop takes a full second: tkinter marshals the call to the main loop, waits ten
times a hundred milliseconds for a loop that has already exited, then gives up
with "main thread is not in main loop". A whole figure's worth of them stalls
whichever thread happens to collect them for about twenty seconds -- and when
that thread is the shared-memory logger, twenty seconds of the recording is
overwritten by the device before it can be written to file.
"""
import gc
import threading
import time

import matplotlib
import pytest
import tkinter

matplotlib.use("TkAgg", force=True)
from matplotlib import pyplot                                  # noqa: E402
from matplotlib._pylab_helpers import Gcf                      # noqa: E402

from libemg.utils import _release_interactive_plot             # noqa: E402


def _live_tk_objects():
    live = 0
    for obj in gc.get_objects():
        try:
            live += isinstance(obj, (tkinter.Variable, tkinter.Image))
        except ReferenceError:
            # A dead weak proxy caught mid-sweep. Not a tkinter object either way.
            continue
    return live


def _seconds_to_collect_on_a_worker_thread():
    elapsed = []

    def worker():
        started = time.perf_counter()
        gc.collect()
        elapsed.append(time.perf_counter() - started)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    return elapsed[0]


def _show_a_plot(close=True):
    """Stand in for one of the ODH visualizers: build, show, close.

    The closure over the figure is deliberate -- the real update callbacks all
    capture it, and that is what keeps the window alive until the frame goes.
    """
    try:
        figure, axis = pyplot.subplots()
    except Exception as error:                  # pragma: no cover
        # No display, or Tcl declining to re-initialize after an earlier
        # interpreter in this process was torn down. Neither says anything about
        # the helper under test.
        pytest.skip(f"could not open a Tk figure: {error}")
    axis.plot([1, 2, 3])
    figure.canvas.draw()

    def update(_frame):
        return axis

    Gcf.destroy_fig(figure)                     # what closing the window does
    if close:
        pyplot.close(figure)


@pytest.fixture(autouse=True)
def no_leftovers():
    """Never let one test hand its unfinalized widgets to the next."""
    yield
    pyplot.close("all")
    gc.collect()


def test_a_shown_plot_is_finalized_before_the_call_returns():
    _release_interactive_plot(_show_a_plot)

    assert _live_tk_objects() == 0
    assert _seconds_to_collect_on_a_worker_thread() < 1.0


def test_a_shown_plot_leaves_enough_widgets_to_matter():
    """What a single figure would cost a worker thread, one second per object."""
    before = _live_tk_objects()
    _show_a_plot(close=False)

    assert _live_tk_objects() - before > 10


def test_finalizing_one_widget_off_the_main_thread_costs_a_second():
    """The per-object price the helper exists to avoid, measured on one object."""
    try:
        figure = pyplot.figure()
    except Exception as error:                  # pragma: no cover
        pytest.skip(f"could not open a Tk figure: {error}")
    master = figure.canvas.get_tk_widget().winfo_toplevel()
    # In a cycle, as matplotlib's own are: refcounting cannot free it, so it
    # waits for whichever thread next runs a collection -- the whole problem.
    holder = {"widget": tkinter.StringVar(master=master, value="x")}
    holder["self"] = holder
    del holder

    stalled = _seconds_to_collect_on_a_worker_thread()

    assert stalled > 0.5, (
        "a worker thread should be the slow place to finalize a Tk object; "
        "if it no longer is, _release_interactive_plot may be obsolete"
    )


def test_the_result_of_the_shown_plot_comes_back():
    assert _release_interactive_plot(lambda: "closed") == "closed"


def test_a_plot_that_fails_to_build_reports_why():
    def show():
        _show_a_plot()
        raise RuntimeError("window failed mid-build")

    with pytest.raises(RuntimeError, match="window failed mid-build"):
        _release_interactive_plot(show)
