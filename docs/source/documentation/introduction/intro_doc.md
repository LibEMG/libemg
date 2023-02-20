Welcome to the LibEMG user guide. This library aims to facilitate the development of EMG-based control systems for offline and online use. In this guide, we will introduce our library, provide a walk-through of many core EMG concepts, and by the end, you will hopefully be ready and excited to explore EMG-based interactions.

# Core Modules
As displayed in Figure 1, this project consists of six main modules. Although many of these modules have been built to stand independently from the others (e.g., Feature Extraction), they work symbiotically together to make up the EMG pipeline. Note that any modules with a dashed border are optional in the context of the pipeline. 

![alt text](core_modules.png)
<center> <p> Figure 1: Diagram of the Core Modules</p> </center>

<h3 style="background-color:#F3FFA8;padding-left: 10px;"> Data Handler </h3>

For offline analysis, loading, parsing, and extracting EMG data can be time-consuming and tedious. Additionally, naming and folder conventions are often unique to specific projects. **For the offline component of this module, the goal is to facilitate this process.** 

All EMG hardware is unique, making it challenging to interface with different devices. **The online component of this module abstracts the hardware complexities and acts as a middleware, listening for EMG data over a UDP socket.** In turn, this module can interface with any device, regardless of its technical specifications, with minor modifications. Additionally, we have built-in support for some common hardware.

<h3 style="background-color:#D3A8FF;padding-left: 10px;"> Filtering </h3>

EMG signals are inherently noisy due to powerline interference, motion artifact, and muscle cross-talk. Filtering EMG is often required to eliminate this noise and improve classification results. **The goal of this module is to provide programmers with a means to easily apply their desired filters.** This module stands independently from the others and can be leveraged for filtering any raw data.

<h3 style="background-color:#FFDCA8;padding-left: 10px;"> Feature Extraction </h3>

As EMG signals are stochastic in nature, they do not provide adequate descriptive information for classification. Feature extraction increases the information density of the underlying signal by computing descriptive properties from a sequence of raw data. **The goal of this module is to provide programmers with a common set of validated features for their projects.** This module stands independently from the others and can be leveraged for any feature extraction task.

<h3 style="background-color:#A8DAFF;padding-left: 10px;"> Feature Selection </h3>

Feature selection is crucial when developing EMG-based control systems as features can often drastically influence performance. Often, however, it is difficult to know what features to select for a particular problem. **The goal of this module is to provide programmers with a means to extract the most relevant features for their specific problem.**

<h3 style="background-color:#A8FFB1;padding-left: 10px;"> Classification </h3>

Classification involves taking in EMG data and making predictions using a particular machine learning model. **This module enables both online (real-time) and offline classification.** Currently, however, it is limited to continuous control schemes.

<h3 style="background-color:#FFA8AD;padding-left: 10px;"> Evaluation </h3>

System evaluation is a crucial part of evaluating the performance of any EMG control system. The two evaluation techniques are **offline** and **online** evaluation. **This module provides programmers with a means to extract common offline evaluation metrics.** Online evaluation is application dependent, so this library does not directly address this issue. This module stands independently from the others and can be leveraged for any offline evaluation tasks.

# Contributing
The repo is open-sourced and can be found [here](https://github.com/eeddy/libemg). For any bugs, improvements, or suggestions please create an issue and we will review it as soon as possible.

# Citing
We ask that if you leverage this library for any research related purposes please cite the following publication:
```
Citation
```