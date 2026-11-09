"""Regression coverage for latest features retained during the feature rebase."""
from types import SimpleNamespace

import numpy as np
import pytest

from libemg import datasets
from libemg._gui import _data_collection_panel as collection
from libemg.utils import get_windows


def test_channel_last_windows_keep_the_upstream_layout():
    samples = np.arange(36).reshape(12, 3)
    normal = get_windows(samples, 6, 2)
    channel_last = get_windows(samples, 6, 2, channel_last=True)
    np.testing.assert_array_equal(channel_last, normal.transpose(0, 2, 1))
    np.testing.assert_array_equal(channel_last[1], samples[2:8])


def test_latest_epn100_dataset_remains_registered():
    assert datasets.get_dataset_list()['EMGEPN100'] is datasets.EMGEPN100
    assert datasets.get_dataset_list(cross_user=True)['EMGEPN100'] is datasets.EMGEPN100


@pytest.mark.parametrize('fail_render', [False, True])
def test_discrete_collection_starts_on_spacebar_and_always_drains_the_logger(monkeypatch, fail_render):
    keys = iter([False, True, True, False])
    events = []
    monkeypatch.setattr(collection, 'dpg', SimpleNamespace(
        mvKey_Spacebar=1,
        is_key_down=lambda key: next(keys),
        set_value=lambda *args, **kwargs: None,
        set_item_width=lambda *args, **kwargs: None,
    ))
    monkeypatch.setattr(collection, 'set_texture', lambda *args, **kwargs: None)
    monkeypatch.setattr(collection.time, 'sleep', lambda seconds: None)
    handler = SimpleNamespace(
        reset=lambda: events.append('reset'),
        start_log=lambda **kwargs: events.append(('start', kwargs)),
        stop_log=lambda: (events.append('stop') or {'emg': 5}),
        modalities=['emg'],
    )
    panel = collection.DataCollectionPanel(handler, discrete=True, timestamps=True)
    def advance(elapsed):
        if fail_render:
            raise RuntimeError('render failed')
    media = SimpleNamespace(fps=24, advance_to=advance,
                            get_dpg_formatted_texture=lambda **kwargs: np.zeros((2, 2, 4)))
    if fail_render:
        with pytest.raises(RuntimeError, match='render failed'):
            panel.play_collection_visual_discrete((media, 'gesture'), 'session_')
    else:
        panel.play_collection_visual_discrete((media, 'gesture'), 'session_')
    assert events == ['reset', ('start', {'file_path': 'session_', 'timestamps': True}), 'stop']