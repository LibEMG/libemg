from unittest.mock import Mock

import libemg.streamers as streamers


def test_mock_emg_stream_starts_background_process(monkeypatch):
    process = Mock()
    process_type = Mock(return_value=process)
    monkeypatch.setattr(streamers, "Process", process_type)

    streamers.mock_emg_stream(
        "samples.csv",
        num_channels=8,
        sampling_rate=200,
        port=54321,
        ip="127.0.0.2",
    )

    process_type.assert_called_once_with(
        target=streamers._stream_mock_emg,
        args=("samples.csv", 8, 200, 54321, "127.0.0.2"),
        daemon=True,
    )
    process.start.assert_called_once_with()
