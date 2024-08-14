from libemg.streamers import new_streamer
from libemg.data_handler import OnlineDataHandler

if __name__ == "__main__":
    """
    This will test your new streamer by plotting it.
    """
    _, sm = new_streamer()
    odh = OnlineDataHandler(sm)
    odh.visualize()


