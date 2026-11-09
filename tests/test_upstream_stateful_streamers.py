"""Exercise upstream device writers against the new published state contract."""
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from libemg.shared_memory_manager import SharedMemoryManager, assign_shared_memory_locks
from libemg._streamers import _mindrove, _OTB_MuoviPlus


@pytest.fixture
def memory():
    tag = 'upstream_' + uuid4().hex
    smm = SharedMemoryManager()
    for item in assign_shared_memory_locks([[tag, (8, 2), np.double],
                                           [tag + '_count', (1, 1), np.int32]]):
        smm.create_variable(*item)
    try:
        yield smm, tag
    finally:
        smm.cleanup(parent=True)


def test_muovi_batch_keeps_its_buffer_order_and_publishes_the_batch(memory):
    smm, tag = memory
    writer = _OTB_MuoviPlus.OTBMuoviPlusEMGStreamer.__new__(_OTB_MuoviPlus.OTBMuoviPlusEMGStreamer)
    writer.smm = SimpleNamespace(commit=lambda name, rows: smm.commit(tag, rows))
    packets = np.array([[10., 20.], [30., 40.], [50., 60.]])
    writer._write_emg_data(packets)
    np.testing.assert_array_equal(smm.get_variable(tag)[:3], packets)
    assert smm.snapshot(tag).total_samples == 3
    assert smm.get_variable(tag + '_count').item() == 3


def test_mindrove_publishes_every_row_returned_by_the_sdk(monkeypatch, memory):
    smm, tag = memory
    incoming = np.array([[10., 20.], [30., 40.]])
    device = SimpleNamespace(
        prepare_session=lambda: None,
        start_stream=lambda: None,
        get_emg_channels=lambda board_id: [0, 1],
        is_prepared=lambda: False,
    )
    monkeypatch.setattr(_mindrove, 'BoardShim', lambda board_id, params: device)
    writer = _mindrove.MindroveStreamer([])
    def read_board(**kwargs):
        writer._stop_event.set()
        return incoming.T
    device.get_board_data = read_board
    adapter = SimpleNamespace(
        commit=lambda name, rows: smm.commit(tag, rows), cleanup=lambda: None,
    )
    monkeypatch.setattr(_mindrove, 'SharedMemoryManager', lambda **kwargs: adapter)
    writer.run()
    np.testing.assert_array_equal(smm.get_variable(tag)[:2], incoming[::-1])
    assert smm.snapshot(tag).total_samples == 2
    assert smm.get_variable(tag + '_count').item() == 2