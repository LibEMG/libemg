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

This module has three data-related functions: **(1) Datasets**, **(2) Offline Data Handling**, and **(3) Online Data Handling**. The dataset functionality has built-in support for downloading and exploring a variety of validated datasets. The OfflineDataHandler is responsible for parsing through previously collected (offline) datasets and preparing data to be passed through the pipeline. Finally, the OnlineDataHandler is responsible for accumulating, storing, and processing real-time data. 

# Datasets
Several validated datasets consisting of different gestures and recording technology are included in this library. These datasets can be used for exploring the library's capabilities and for future research. When using the packaged datasets for research purposes, we ask that you reference the original dataset contribution and not just this toolkit (the original dataset contributions might not be obvious since they are included for download with this toolkit).

<details>
<summary><b>OneSubjectMyoDataset</b></summary>

| Attribute          | Description |
| ------------------ | ----------- |
| **Num Subjects:**      | 1       |
| **Num Reps:**      | 12 Reps (i.e., 6 Trials x 2 Reps)|
| **Time Per Rep:**      | 3s      |
| **Classes:**       | <ul><li>0 - Hand Open</li><li>1 - Hand Close</li><li>2 - No Movement</li><li>3 - Wrist Extension</li><li>4 - Wrist Flexion</li></ul>       |
| **Device:**        | Myo        |
| **Sampling Rates:** | EMG (200 Hz)        |

**Using the Dataset:**
```Python
from libemg.datasets import OneSubjectMyoDataset
dataset = OneSubjectMyoDataset(redownload=False)
odh = dataset.prepare_data()
```

**References:**
```
Work to be published...
```
-------------
</details>

<br>

<details>
<summary><b>3DCDatset</b></summary>

| Attribute          | Description |
| ------------------ | ----------- |
| **Num Subjects:**      | 22       |
| **Num Reps:**      | 4 Training, 4 Testing       |
| **Time Per Rep:**      | 5s      |
| **Classes:**       | <ul><li>0 - No Motion</li><li>1 - Radial Deviaton</li><li>2 - Wrist Flexion</li><li>3 - Ulnar Deviaton</li><li>4 - Wrist Extension</li><li>5 - Supination</li><li>6 - Pronation</li><li>7 - Power Grip</li><li>8- Open Hand</li><li>9 - Chuck Grip</li><li>10 - Pinch Grip</li></ul>       |
| **Device:**        | Delsys        |
| **Sampling Rates:** | EMG (1000 Hz)        |

**Using the Dataset:**
```Python
from libemg.datasets import _3DCDataset
dataset = _3DCDataset(redownload=False)
odh = dataset.prepare_data()
```

**References:**
```
@article{cote2019deep, title={Deep learning for electromyographic hand gesture signal classification using transfer learning}, author={C{^o}t{'e}-Allard, Ulysse and Fall, Cheikh Latyr and Drouin, Alexandre and Campeau-Lecours, Alexandre and Gosselin, Cl{'e}ment and Glette, Kyrre and Laviolette, Fran{\c{c}}ois and Gosselin, Benoit}, journal={IEEE transactions on neural systems and rehabilitation engineering}, volume={27}, number={4}, pages={760--771}, year={2019}, publisher={IEEE} }

@article{cote2020interpreting, title={Interpreting deep learning features for myoelectric control: A comparison with handcrafted features}, author={C{^o}t{'e}-Allard, Ulysse and Campbell, Evan and Phinyomark, Angkoon and Laviolette, Fran{\c{c}}ois and Gosselin, Benoit and Scheme, Erik}, journal={Frontiers in Bioengineering and Biotechnology}, volume={8}, pages={158}, year={2020}, publisher={Frontiers Media SA} }
```
-------------
</details>

<br/>

<details>
<summary><b>Nina Pro DB2</b></summary>

<br/>
The Ninapro DB2 is a dataset that can be used to test how algorithms perform for large gesture sets. The dataset contains 6 repetitions of 50 motion classes (plus optional rest) that were recorded using 12 Delsys Trigno electrodes around the forearm.

<br/>
<br/>

Note, this dataset will not be automatically downloaded. To download this dataset, please see [Nina DB2](http://ninapro.hevs.ch/node/17). Simply download the ZIPs and place them in a folder and LibEMG will handle the rest. All credit for this dataset should be given to the original authors. 

<br/>

| Attribute          | Description |
| ------------------ | ----------- |
| **Num Subjects:**      | 40       |
| **Num Reps:**      | 6 |
| **Time Per Rep:**      | 5s      |
| **Classes:**       | 50 [Nina Pro DB2](http://ninapro.hevs.ch/node/123)    |
| **Device:**        | Delsys        |
| **Sampling Rates:** | EMG (2000 Hz)        |

**Using the Dataset:**
```Python
from libemg.datasets import NinaproDB2
dataset = NinaproDB2("data/NinaDB2") #The loacation of Nina DB2 is downloaded
odh = dataset.prepare_data()
```

**References:**
```
Atzori, M., Gijsberts, A., Castellini, C. et al. 
Electromyography data for non-invasive naturally-controlled robotic hand prostheses. 
Sci Data 1, 140053 (2014). 
https://doi.org/10.1038/sdata.2014.53
```
-------------

</details>
</br>

<details>
<summary><b>Nina Pro DB8</b></summary>

<br/>

Note, this dataset will not be automatically downloaded. To download this dataset, please see [Nina DB8](http://ninapro.hevs.ch/DB8). Simply download the ZIPs and place them in a folder and LibEMG will handle the rest. All credit for this dataset should be given to the original authors. 

<br/>

| Attribute          | Description |
| ------------------ | ----------- |
| **Num Subjects:**      | 12       |
| **Num Reps:**      | 20 Training, 2 Testing |
| **Time Per Rep:**      | 6-9s      |
| **Classes:**       | 9 [Nina Pro DB8](http://ninapro.hevs.ch/DB8)    |
| **Device:**        | Delsys        |
| **Sampling Rates:** | EMG (1111 Hz)        |

**Using the Dataset:**
```Python
from libemg.datasets import NinaproDB8
dataset = NinaproDB8("data/NinaDB8") #The loacation of Nina DB8 is downloaded
odh = dataset.prepare_data()
```

**References:**
```
AUTHOR=Krasoulis Agamemnon, Vijayakumar Sethu, Nazarpour Kianoush
TITLE=Effect of User Practice on Prosthetic Finger Control With an Intuitive Myoelectric Decoder  
JOURNAL=Frontiers in Neuroscience     
VOLUME=13      
YEAR=2019   
URL=https://www.frontiersin.org/articles/10.3389/fnins.2019.00891     
DOI=10.3389/fnins.2019.00891    	
ISSN=1662-453X   
```
-------------

</details>
</br>

# Offline Data Handler 
One overhead for most EMG projects is interfacing with a particular dataset since they often have different folder and file structures. LibEMG provides a means to quickly interface datasets so you can focus on using them with minimal setup time. Assuming the files in the dataset are well formatted (i.e., they include all metadata such as rep, class, and subject) and are either .csv or .txt files, the OfflineDataHandler does all accumulation and processing. To do this, LibEMG relies on regular expressions to define a dataset's file and folder structure. These expressions can be used to create a dictionary that is passed to the OfflineDataHandler. Once the data handler has collected all the files that satisfy the regexes, the dataset can be sliced using the metadata tags (e.g., by rep, subjects, classes, etc.). After extracting the data it is ready to be passed through the rest of the pipeline. The following code snippet exemplifies how to process a dataset with testing/training, rep, and class metadata. In this case the file format is: `dataset/train/R_1_C_1_EMG.csv` where R is the rep and C is the class.

<details>
<summary><b>Example Code</b></summary>

```Python
from libemg.data_handler import OfflineDataHandler, RegexFilter
dataset_folder = 'dataset'
regex_filters = [
    RegexFilter(left_bound = "dataset/", right_bound="/", values = sets_values, description='sets'),
    RegexFilter(left_bound = "_C_", right_bound="_EMG.csv", values = classes_values, description='classes'),
    RegexFilter(left_bound = "R_", right_bound="_C_", values = reps_values, description='reps')
]
odh = OfflineDataHandler()
odh.get_data(folder_location=dataset_folder, regex_filters=regex_filters, delimiter=",")

# Extract training data:
train_odh = odh.isolate_data(key="sets", values=[0])
train_windows, train_metadata = train_odh.parse_windows(50,25)

# Extract features
fe = FeatureExtractor()
feature_list = fe.get_feature_list()
training_features = fe.extract_features(feature_list, train_windows)
```

</details>
</br>

# Online Data Handler 

One complication when using EMG devices is the lack of standardization, meaning that interfacing with hardware is a new undertaking for each device. A goal of LibEMG is to abstract these differences and enable a hardware-agnostic framework. Therefore, this module acts as a middle layer for processing real-time data streaming from any device. In this architecture - exemplified in Figure 1 – live data streaming is performed by using a shared memory buffer as the core. This shared memory buffer is created by the device streamer, where a process is spawned that continuously populates the buffer with samples. Other modules can gain access to the shared memory buffer using the shared memory items that the streamer returned, allowing for cross-process, low-latency, non-blocking access to the data of interest. We provide an OnlineDataHandler object that is a generic object for attaching to the shared memory buffer with added some utilities.
 
An example of the online data streaming workflow is provided below:
 
```Python
# start the hardware streamer, receive a process handle and shared memory descriptors
streamer_process, shared_memory_items = libemg.streamers.myo_streamer()
# start an online data handler to attach to the shared memory buffer
odh = libemg.data_handler.OnlineDataHandler(shared_memory_items=shared_memory_items)
# start logging the data to a file
odh.log_to_file()
# start visualizing the data
odh.visualization()
```

![alt text](online_dh.png)
<center> <p> Figure 1: OnlineDataHandler Architecture</p> </center>

**For more information on the default streamers and creating your own, please reference the Supported Hardware section.** 
