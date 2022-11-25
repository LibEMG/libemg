import time
import pytest
import numpy as np
from unb_emg_toolbox.utils import mock_emg_stream
from unb_emg_toolbox.data_handler import OnlineDataHandler

# def test_mock_stream_500HZ():
#     online_data_handler = OnlineDataHandler(emg_arr=True)
#     online_data_handler.start_listening()
#     mock_emg_stream("tests/data/stream_data_tester.csv", num_channels=8, sampling_rate=500)
#     while len(online_data_handler.raw_data.get_emg()) < 2000:
#         if len(online_data_handler.raw_data.get_emg()) == 1:
#             s_time = time.time()
#     online_data_handler.stop_data()
#     assert np.abs(4 - (time.time() - s_time)) < 0.1 # Giving 0.1 second room for error

# def test_mock_stream_1000HZ():
#     online_data_handler = OnlineDataHandler(emg_arr=True)
#     online_data_handler.start_listening()
#     mock_emg_stream("tests/data/stream_data_tester.csv", num_channels=8, sampling_rate=1000)
#     while len(online_data_handler.raw_data.get_emg()) < 2000:
#         if len(online_data_handler.raw_data.get_emg()) == 1:
#             s_time = time.time()
#     online_data_handler.stop_data()
#     assert np.abs(2 - (time.time() - s_time)) < 0.1 # Giving 0.1 second room for error

@pytest.mark.skip(reason="Doesn't Work in CI")
def test_mock_stream_2000HZ():
    online_data_handler = OnlineDataHandler(emg_arr=True)
    online_data_handler.start_listening()
    s_time = None
    mock_emg_stream("tests/data/stream_data_tester.csv", num_channels=8, sampling_rate=2000)
    while len(online_data_handler.raw_data.get_emg()) < 2000:
        if len(online_data_handler.raw_data.get_emg()) > 0 and s_time is None:
            s_time = time.time()
    online_data_handler.stop_data()
    assert np.abs(1 - (time.time() - s_time)) < 0.1 # Giving 0.1 second room for error
