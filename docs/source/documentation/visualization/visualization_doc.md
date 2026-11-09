# Online Data Handler
There is different convenient functionality in the `OnlineDataHandler` to help developers. Firstly, two visualize functions: `visualize` and `visualize_channels`, exemplified in Table 1 are included.

```Python
from libemg.streamers import myo_streamer
from libemg.data_handler import OnlineDataHandler

if __name__ == "__main__":
    streamer, sm = myo_streamer()
    odh = OnlineDataHandler(sm)
    odh.visualize()
```

| <center>Combined</center>  | <center>All Channels</center> |
| ------------- | ------------- |
| ![alt text](all_channels.gif)  | ![alt text](multi_channel.gif)   |
<center> <p> Figure 1: Raw Data from the <b>OnlineDataHandler</b></p> </center>

These plots redraw on a timer, and each frame copies the whole buffer out of shared memory. For a cheaper look at live data, install a probe hook instead. A `ProbeHook` carries a `Periodic(hz)` criterion and declares no outputs, so it is woken at most `hz` times a second and nothing downstream waits on it. A probe cannot stall or alter the pipeline it watches, which makes it the safe way to observe a running control system.

```Python
from libemg.reactive import ProbeHook

odh.install_hook(ProbeHook('watch', 'emg', print, hz=30))
odh.start_hooks()
...
odh.stop_hooks()
```

Pass your own function in place of `print` to draw, log, or forward the samples. See the Reactive Pipelines section for the other criteria and hooks.

# EMG Classifier 
The EMG classifier contains a visualization tool for viewing the decisions stream (i.e., the predictions over time) for a particular classifier using the `visualize` function. 


![alt text](decision_stream.png)
<center> <p> Figure 2: The decision stream of a classifier.</b></p> </center>

# EMG Regressor

The EMG regressor also contains a visualization tool for viewing the model's decision stream. Similar to the classifier, you can view the decision stream using the `visualize` method.

![alt text](regressor.png)
<center> <p> Figure 3: The decision stream of a regressor.</b></p> </center>

# Feature Extractor 
The Feature Extrator and Online Data Handler contain a visualization tool for viewing the PCA feature space. This can be done using the `visualize_feature_space` function. If this function is run on an online data handler, a live PCA feature space will be shown (see Figure 4).

| <center>Offline</center>  | <center>Online (Live)</center> |
| ------------- | ------------- |
| ![alt text](feature_space.png)  | ![alt text](feature_space.gif) |
<center> <p> Figure 4: The PCA feature space of a set of data.</p> </center>

# Filtering 
The filtering module has a `visualize_effect` function that demonstrates the effect of a filter on a set of data in the time and frequency domain.

![](filtering_1.png)
<center> <p> Figure 5: Data before and after filtering in the time and frequency domain.</p> </center>

# Heatmap

Viewing EMG as a time series may not be appropriate for high-density EMG systems. `LibEMG` offers a live heatmap visualization using the `visualize_heatmap` method. Heatmaps of multiple features can be visualized in real-time to show spatial information (only features that produce a single value per window are supported).

![alt text](heatmap.gif)
<center> <p> Figure 6: Real-time heatmap visualization.</p> </center>
