<style>
    table {
        width: 100%;
    }
    .device_img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
        height: 50%;
    }
    .device_img_2 {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 35%;
        height: 50%;
    }
</style>

By default, LibEMG supports several hardware devices (shown in Table 1). 
- The **Myo Armband** is a previously available commercial device popular for HCI applications due to its low cost.
- The [**Delsys**](https://delsys.com/) is a commercially available system primarily used for medical applications due to its relatively high cost. 
- The [**SIFI Cuff**](https://sifilabs.com/) is a pre-released device that will soon be commercially available. Compared to the Myo armband, this device has a much higher sampling rate (~2000 Hz).
- The [**Oymotion Cuff**](http://www.oymotion.com/en/product32/149) is a commercial device that samples EMG at 1000 Hz (8 bits) or 500 Hz (12 bits). 
- The [**OTBioelettronica**](https://otbioelettronica.it/hardware/) devices are a set of commercially available HDEMG systems.

If selecting EMG hardware for real-time use, wireless armbands that sample above 500 Hz are preferred. Additionally, future iterations of LibEMG will include Inertial Measurement Unit (IMU) support. As such, devices should have IMUs to enable more interaction opportunities.

| <center>Hardware</center> | <center>Function</center> | <center>Image</center> |
| ------------- | ------------- | ------------- |
| Myo Armband  | `myo_streamer()`  | <div class="device_img">![](devices/Myo.png) </div>|
| Delsys  | `delsys_streamer()` or `delsys_API_streamer()` | <div class="device_img_2">![](devices/delsys_trigno.png) </div>|
| SIFI Cuff | `sifi_streamer()` | <div class="device_img">![](devices/sifi_cuff.png) </div>|
| Oymotion | `oymotion_streamer()`| <div class="device_img">![](devices/oymotion.png) </div>|
| Muovi | `otb_muovi_streamer()`| <div class="device_img">![](devices/muovi.png) </div>| 
| Muovi+ | `otb_muovi_plus_streamer()`| <div class="device_img">![](devices/muovi+.png) </div>| 
| Sessantaquattro+ | `otb_sessantaquattro_plus_streamer()`| <div class="device_img">![](devices/sess.png) </div>| 

<center> <p>Table 1: The list of all implemented streamers.</p> </center>

## Inspecting Hardware
LibEMG includes an `analyze_hardware` function to run an analysis on the hardware device being used. This is a good way to  check that your device is working as expected. Example output from this function for the Myo Armband is:

```Python
from libemg.streamers import myo_streamer
from libemg.data_handler import OnlineDataHandler

if __name__ == "__main__":
    streamer, sm = myo_streamer()
    odh = OnlineDataHandler(sm)
    odh.analyze_hardware()
```

```
Starting analysis (10s)... We suggest that you elicit varying contractions and intensities to get an accurate analysis.
Sampling Rate: 200
Num Channels: 8
Max Value: 127
Min Value: -128
Resolution: 8 bits
Repeating Values: 0
```

- `sampling rate` is the number of samples read per second.
- `num channels` is the number of channels being read.
- `min and max values` are the maximum values read from the data.
- `resolution` is the predicted resolution in bits based on the data seen.
- `repeating values` is the number of repeated values in the input data. Repeating values that are > 0 indicate that there might be some issues (e.g., the hardware sensor is malfunctioning).

## Creating Custom Streamers
Custom UDP streamers can be created to interface with other hardware. A UDP streamer reads a value from a device, pickles it, and sends it over UDP. An example streamer for the Myo Armband is shown in the code snippet below.

<details>
<summary><b>Example Code</b></summary>

```Python
import socket
import multiprocessing
import pickle
from pyomyo import Myo, emg_mode

def streamer():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()

    # On every new sample it, simply pickle it and write it over UDP
    def write_to_socket(emg, movement):
        data_arr = pickle.dumps(list(emg))
        sock.sendto(data_arr, ('127.0.0.1' 12345))
    m.add_emg_handler(write_to_socket)
    
    while True:
        m.run()
        
if __name__ == "__main__" :
    # Create streamer in a seperate Proces so that the main thread is free
    p = multiprocessing.Process(target=streamer, daemon=True)
    p.start()
    
    # Code leveraging the data goes here:
    odh = OnlineDataHandler(emg_arr=True, port=12345, ip='127.0.0.1')
    odh.start_listening()

    # Do stuff with data...
```

</details>

<br/>
<br/>