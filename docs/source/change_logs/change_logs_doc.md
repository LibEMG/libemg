<style>
.data_handler {
    background-color:#F3FFA8;
    padding-left: 10px;
}
.filtering {
    background-color:#D3A8FF;
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
</style>

**Legend:**
- <span class="enhancement"> New Feature </span>
- <span class="api_change"> Api Change </span> 
- <span class="fix"> Fix </span> 

# V 1.0.0 
**November 2022**

This is the first release of the UNB EMG Toolbox! Thanks to everyone who has made this possible, and we look forward to seeing where this goes.

<h2 style="background-color:#B1B1B1;padding-left: 10px;"> Core Module Changes </h2>

As part of V 1.0.0 we have included 7 main core modules: data handling, filtering, feature extraction, feature selection, classification, and evaluation.

<h3 class="data_handler"> Data </h3>

Facilitating the process of offline and online data processing. Additionally, adding a number of datasets.

- <span class="enhancement"> New Feature </span>**OfflineDataHandler (class):** Process offline data.
- <span class="enhancement"> New Feature </span>**OnlineDataHandler (class):** Process online (real-time) data.
- <span class="enhancement"> New Feature </span>**Datasets:** _3DCDataset, OneSubjectMyoDataset

<h3 class="filtering"> Filtering </h3>

Applying filters to raw EMG data for eliminating common sources of noise/contamination.

- <span class="enhancement"> New Feature </span>**Filter (class):** Facilitates the process of filtering raw emg data. Works for both online and offline data. 