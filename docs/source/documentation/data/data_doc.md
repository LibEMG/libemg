This module has three main pieces of data-related functionality: **(1) Datasets**, **(2) Offline Data Handling**, and **(3) Online Data Handling**. Together, they correspond to most of the data-related functionality one would ever need for leveraging validated datasets, parsing through file structures of offline data, and processing real-time EMG.

# EMG Datasets
To enable all interested parties the ability to leverage our toolkit, we have included several validated datasets as part of this toolkit. These datasets can be leveraged for exploring the toolkit's capabilities and, additionally, for future research. We ask that for the latter, please correctly reference the dataset in your work. Note, these datasets are stored on github and will be cloned locally when downloaded. 

## 3DCDataset
Short Description goes here.

<style>
    table {
        width: 100%;
    }
</style>
| Attribute          | Description |
| ------------------ | ----------- |
| **Num Reps:**      | 4 Training, 4 Testing       |
| **Time Per Rep:**      | 5s      |
| **Classes:**       | <ul><li>0 - No Motion</li><li>1 - Radial Deviaton</li><li>2 - Wrist Flexion</li><li>3 - Ulnar Deviaton</li><li>4 - Wrist Extension</li><li>5 - Supination</li><li>6 - Pronation</li><li>7 - Power Grip</li><li>8- Open Hand</li><li>9 - Chuck Grip</li><li>10 - Pinch Grip</li></ul>       |
| **Device:**        | Delsys        |
| **Sampling Rates:** | EMG (1000 Hz)        |
| **Continuous:**    | False |
| **Repo:**          | https://github.com/ECEEvanCampbell/3DCDataset |

### Using the Dataset 
```Python
dataset = _3DCDataset(save_dir='3dc_data', redownload=False)
odh = dataset.prepare_data(format=OfflineDataHandler)
```

### References
```
@article{cote2019deep, title={Deep learning for electromyographic hand gesture signal classification using transfer learning}, author={C{^o}t{'e}-Allard, Ulysse and Fall, Cheikh Latyr and Drouin, Alexandre and Campeau-Lecours, Alexandre and Gosselin, Cl{'e}ment and Glette, Kyrre and Laviolette, Fran{\c{c}}ois and Gosselin, Benoit}, journal={IEEE transactions on neural systems and rehabilitation engineering}, volume={27}, number={4}, pages={760--771}, year={2019}, publisher={IEEE} }

@article{cote2020interpreting, title={Interpreting deep learning features for myoelectric control: A comparison with handcrafted features}, author={C{^o}t{'e}-Allard, Ulysse and Campbell, Evan and Phinyomark, Angkoon and Laviolette, Fran{\c{c}}ois and Gosselin, Benoit and Scheme, Erik}, journal={Frontiers in Bioengineering and Biotechnology}, volume={8}, pages={158}, year={2020}, publisher={Frontiers Media SA} }
```

## OneSubjectMyoDataset
This is a simple one-subject dataset used for some of the examples. It includes the pre-filtered EMG data streamed from the Myo Armaband for four different contractions. Each was performed while the arm was resting on the armrest of a chair. 

<style>
    table {
        width: 100%;
    }
</style>
| Attribute          | Description |
| ------------------ | ----------- |
| **Num Reps:**      | 5 Training, 3 Testing       |
| **Time Per Rep:**      | 5s      |
| **Classes:**       | <ul><li>0 - Hand Closed</li><li>1 - Hand Open</li><li>2 - No Movement</li><li>3 - Wrist Extension</li><li>4 - Wrist Flexion</li></ul>       |
| **Device:**        | Myo Armband      |
| **Sampling Rates:** | EMG (200 Hz)        |
| **Continuous:**    | False |
| **Repo:**          | https://github.com/eeddy/OneSubjectMyoDataset |

### Using the Dataset 
```Python
dataset = OneSubjectMyoDataset(save_dir='one_subject_myo', redownload=False)
odh = dataset.prepare_data(format=OfflineDataHandler)
```

# Offline Data Handler 
TODO: Evan

# Online Data Handler 
One of the major complications in interfacing with EMG devices is that they are all unique. The thing that they all share in common, however, is that they sample EMG at a specific frequency. To handle these differences, we have decided to abstract the device out of the toolkit, and create a middle layer level for processing data from any device instead. In this architecture - exemplified in Figure 1 - the online data handler reads data from a UDP port. Once data is read, it is passed through the system and is processed equivalently for any hardware. While this means that developers must create their custom UDP streamer, it is often quite simple. For example, Figure 2 shows how simple this implementation is for the Myo Armband. The advantage of this architecture, is that it enables our toolkit to interface with any device at any sampling rate as long as data can be streamed over UDP. 

![alt text](online_dh.png)
<center> <p> Figure 1: Online Data Handler Architecture</p> </center>

```Python
import socket
import multiprocessing
from pyomyo import Myo, emg_mode

def streamer():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    m = Myo(mode=emg_mode.FILTERED)
    m.connect()

    def write_to_socket(emg, movement):
        sock.sendto(bytes(str(emg), "utf-8"), ('127.0.0.1', 12345))
    m.add_emg_handler(write_to_socket)
    
    while True:
        m.run()
        
if __name__ == "__main__" :
    # Create streamer in a seperate Proces:
    p = multiprocessing.Process(target=streamer, daemon=True)
    p.start()
    
    # Code leveraging the data goes here:
    odh = OnlineDataHandler(emg_arr=True, port=12345, ip='127.0.0.1')
    odh.get_data()

    # Do stuff with data...
```
<center> <p> Figure 2: Example of an EMG Streamer for the Myo Armband</p> </center>
