<style>
.data_handler {
    background-color:#F3FFA8;
    padding-left: 10px;
}
.filtering {
    background-color:#D3A8FF;
    padding-left: 10px;
}
.feature_extraction {
    background-color:#FFDCA8;
    padding-left: 10px;
}
.feature_selection {
    background-color:#A8DAFF;
    padding-left: 10px;
}
.classification {
    background-color:#A8FFB1;
    padding-left: 10px;
}
.offline_metrics {
    background-color:#FFA8AD;
    padding-left: 10px;
}
.enhancement {
    background-color:green;
    width:130px;
    padding-left:10px;
    padding-right:10px;
    border-radius: 25px;
    margin-right: 10px;
}
.api_change {
    background-color:yellow;
    width:110px;
    padding-left:10px;
    padding-right:10px;
    border-radius: 25px;
    margin-right: 10px;
}
.fix {
    background-color:red;
    width:45px;
    padding-left:10px;
    padding-right:10px;
    border-radius: 25px;
    margin-right: 10px;
}
.grey_header {
    background-color:#B1B1B1;
    padding-left: 10px;
}
</style>

**Legend:**
- <span class="enhancement"> New Feature </span>
- <span class="api_change"> Api Change </span> 
- <span class="fix"> Fix </span> 

---------------------
# V 1.0.0 
**December 2022**

This is the first release of the UNB EMG Toolbox! Thanks to everyone who has made this possible, and we look forward to seeing where this goes.

<h2 class="grey_header"> Core Module Changes </h2>

As part of V 1.0.0 we have included 7 main core modules: data handling, filtering, feature extraction, feature selection, classification, and evaluation. In addition to the core modules, we have included two sub-modules: utils, and screen guided training.

<h3 class="data_handler"> Data </h3>

Facilitating the process of offline and online data processing. Additionally, adding a number of datasets.

- <span class="enhancement"> New Feature </span>**OfflineDataHandler (class):** Process offline data.
- <span class="enhancement"> New Feature </span>**OnlineDataHandler (class):** Process online (real-time) data.
- <span class="enhancement"> New Feature </span>**Datasets:** _3DCDataset, OneSubjectMyoDataset

<h3 class="filtering"> Filtering </h3>

Applying filters to raw EMG data for eliminating common sources of noise/contamination.

- <span class="enhancement"> New Feature </span>**Filter (class):** Facilitates the process of filtering raw emg data. Works for both online and offline data. 

<h3 class="feature_extraction"> Feature Extraction </h3>

- <span class="enhancement"> New Feature </span>**FeatureExtractor (class):** Extract ~X different EMG features and ~X feature groups.

- **Features:**
- **Feature Groups:**
<!-- TODO Fix the X's -->

<h3 class="feature_selection"> Feature Selection </h3>

- <span class="enhancement"> New Feature </span>**FeatureSelector (class):** Select the optimal features based on a certain metric.

- **Metrics:** accuracy, active error, mean semi principle axis, feature efficiency, repeatibility index, and seperability index

<h3 class="classification"> Classification </h3>

- <span class="enhancement"> New Feature </span>**Online/OfflineEMGClassifier (class):** Create an online or offline classifier.

- **Classifiers**: LDA, SVM, MLP, RF, NB, GB, KNN, QDA

<h3 class="offline_metrics"> Offline Evaluation Metrics </h3>

- <span class="enhancement"> New Feature </span>**OfflineMetrics (class):** Evaluate an OfflineEMGClassifier's performance.

- **Metrics:** classification accuracy, active error, instability, confusion matrix, F1, precision, recall, and rejection rate

<h3 class="grey_header"> Utils </h3>

- <span class="enhancement"> New Feature </span>**get_windows (method):** Parses windows from raw emg.
- <span class="enhancement"> New Feature </span>**get_regex (method):** Creates regex for offline data handler.
- <span class="enhancement"> New Feature </span>**mock_emg_stream (method):** Creates a streamer based on a file of EMG data.
- <span class="enhancement"> New Feature </span>**myo_streamer (method):** Streamer for the Myo armband.
- <span class="enhancement"> New Feature </span>**sifi_streamer (method):** Streamer for the SIFI labs cuff.
- <span class="enhancement"> New Feature </span>**delsys_streamer (method):** Streamer for the Delsys.

<h3 class="grey_header"> Screen Guided Training </h3>

- <span class="enhancement"> New Feature </span>**TrainingUI (class):** Training module for acquiring offline training/testing data.

<h3 class="grey_header"> Examples </h3>

- **Example 1:** Snake Game (Myo, Sifi, Delsys)
- **Example 2:** Fitts Task Experiment (Offline vs Online)
- **Example 3:** Unity Game 
- **Example 4:** Deep Learning

--------------------