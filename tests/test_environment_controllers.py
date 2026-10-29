"""The GUI's Classifier controller has to answer everything Fitts asks of it.

Fitts asks every message for its prediction and a timestamp, plus the speed when
proportional control is on (the default). The GUI builds its Classifier
controller in the 'predictions' format, '<class> [speed]', which carries no
timestamp; asking for one used to raise, ending the task on its first message.
"""
import socket
import time

from libemg._gui._environments import ControllerSpec


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _receive(controller, port, message, info):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.sendto(message.encode(), ("127.0.0.1", port))
    deadline = time.time() + 2
    while time.time() < deadline:
        data = controller.get_data(info)
        if data is not None:
            return data
        time.sleep(0.01)
    raise AssertionError("no message arrived")


def test_predictions_format_answers_what_fitts_asks():
    port = _free_port()
    controller = ControllerSpec(kind="classifier", port=port).build()   # as the GUI builds it
    before = time.time()
    predictions, timestamp, speed = _receive(controller, port, "3 0.5",
                                             ["predictions", "timestamp", "pc"])
    assert predictions == [3.0]
    assert speed == [0.5]
    # The format has no timestamp, so it is when the message was read.
    assert before <= timestamp <= time.time()


def test_probabilities_format_still_reads_its_timestamp():
    port = _free_port()
    controller = ControllerSpec(kind="classifier", port=port, num_classes=3,
                                output_format="probabilities").build()
    predictions, timestamp = _receive(controller, port, "0.1 0.7 0.2 1234.5",
                                      ["predictions", "timestamp"])
    assert predictions == [1.0]
    assert timestamp == 1234.5
