import sys

from libemg._streamers import _myo_streamer
from libemg._streamers import _delsys_streamer
from libemg._streamers import _delsys_API_streamer
from libemg._streamers import _emager_streamer
from libemg._streamers import _sifi_bridge_streamer
from libemg._streamers import _OTB_Muovi
from libemg._streamers import _OTB_MuoviPlus
from libemg._streamers import _OTB_SessantaquattroPlus
from libemg._streamers import _OTB_Syncstation
if sys.platform.startswith('win'):
    from libemg._streamers import _oymotion_windows_streamer
else:
    from libemg._streamers import _oymotion_streamer
from libemg._streamers import _emager_streamer
from libemg._streamers import _leap_streamer
