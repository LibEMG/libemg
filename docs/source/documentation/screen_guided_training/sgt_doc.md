<style>
    table {
        width: 100%;
    }
</style>

While not directly part of the core pipeline itself, the `Screen Guided Training` module is a crucial aspect of this project. For those who wish to run data collection studies, or anyone looking to develop online interactions, this module will be helpful. Note that our SGT tool can handle image types of `png`, `jpg` and `jpeg`. Additionally, it can handle `.gifs` for dynamic movements/contractions.

| <center>Menu</center>  | <center>SGT</center> |
| ------------- | ------------- |
| ![](menu.PNG) | ![](training.PNG) |
<center> <p> Table 1: Screen Guided Training UI</p> </center>


# Gesture Library
One of the tedious aspects (and annoyances) of performing data collection is that we often need to accumulate photos (or videos) of each desired contractions. Often, this process is time-consuming and ultimately leads to inconsistencies among studies. To streamline this process, we have created a database of ~35 common gestures (both static and dynamic) that are interfaced directly with this project.

The library can be found here: https://github.com/anon/anon

![](gesture_example.png)
<center> <p> Figure 1: <a href="https://github.com/anon/anon">Gesture Library</a></p> </center>

Each gesture in the library is associated with an ID and can be downloaded using those ids. A simple pipeline for setting up the SGT module is as follows: 

```Python
from libemg.screen_guided_training import ScreenGuidedTraining
from libemg.data_handler import OnlineDataHandler
from libemg.streamers import myo_streamer

# Create data handler and streamer 
odh = OnlineDataHandler(emg_arr=True)
odh.start_listening()
myo_streamer()

train_ui = ScreenGuidedTraining()
# Download gestures with indices 1,2,3,4,5 and store them in the "gestures/" folder
train_ui.download_gestures([1,2,3,4,5], "gestures/", download_gifs=True)
# Launch the training UI
train_ui.launch_training(odh, output_folder="demos/data/sgt/", rep_folder="demos/images/test/")
```