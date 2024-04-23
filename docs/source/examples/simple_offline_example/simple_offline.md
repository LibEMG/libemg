[View Source Code](https://github.com/libemg/LibEMG_OneSubject_Showcase)

<img src="https://github.com/libemg/LibEMG_OneSubject_Showcase/blob/main/Docs/Results.png?raw=true"/>

This simple offline analysis serves as a starting point to understand the offline functionality of the library. In this example, we will load in the OneSubjectMyoDataset and test the classification accuracy of several different classifiers. Finally, we will look at the PCA feature space of the data. All code can be found in `main.py`.

## Step 1: Importing LibEMG
The very first step involves importing the modules needed. In general, each of LibEMG's modules has its own import. Make sure that you have successfully installed libemg through pip.
```Python
from libemg.datasets import OneSubjectMyoDataset
from libemg.emg_predictor import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics
import matplotlib.pyplot as plt
```

## Step 2: Setting up Constants
Window and increment size selection will have major impacts on a control system. In general, while bigger window sizes can improve classification accuracy, they introduce lag to the system as predictions are made on older data. Moreover, smaller increment sizes result in more responsive systems. Ultimately, this comes with increased hardware costs to process these decisions. LibEMG defines window and increment sizes as the number of samples. In this case, the dataset was recorded from the Myo Armband (which samples at 200Hz), so a window of 20 samples corresponds to 100ms. 

Determining the features or feature set to use is another design consideration. Often, it is suggested to use one of the predefined validated feature groups such as HTD, LS4, or LS9.

```Python
WINDOW_SIZE = 20
WINDOW_INCREMENT = 10 
FEATURE_SET = "HTD"
```

# Step 3: Loading in Dataset
In this example, we load in the OneSubjectMyoDataset, which is automatically downloaded and stored in an OfflineDataHandler. After preparing the data set, we isolate it into sets 0-2 and 3-5. Since this dataset consists of six trials, trials 1-3 will be used for training and 4-6 for testing. Finally, we can extract windows from both the training and testing data.

```Python
# Load in dataset
dataset = OneSubjectMyoDataset()
data = dataset.prepare_data()

# Split data into training and testing
train_data = data.isolate_data("sets", [0,1,2]) 
test_data = data.isolate_data("sets", [3,4,5]) 

# Extract windows 
train_windows, train_meta = train_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
test_windows, test_meta = test_data.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
```

# Step 4: Feature Extraction
We first create a data_set dictionary, since it is required to be passed to each classifier. We also extract features from the test windows. 

```Python
# Create data set dictionary using training data
data_set = {}
data_set['training_features'] = fe.extract_feature_group(FEATURE_SET, train_windows)
data_set['training_labels'] = train_meta['classes']

test_features = fe.extract_feature_group(FEATURE_SET, test_windows)
```

# Step 5: Classification
Iterating on a list of predefined classifiers, we can fit a classifier to the predefined dataset, run it to get the predictions on the test data, and then extract offline metrics. This data could then be accumulated and plotted if desired.

```Python
classifiers = ["LDA","SVM","KNN","RF","QDA"]

# Extract metrics for each classifier
for classifier in classifiers:
    model = EMGClassifier(classifier)

    # Fit and run the classifier
    model.fit(data_set.copy())
    preds, probs = model.run(test_features)

    # Null label is 2 since it is the no movement class
    metrics = om.extract_common_metrics(test_meta["classes"], preds, 2)

```

# Step 6: PCA Feature Space Projection
Since LibEMG has a visualize_feature_space function built-in, the PCA feature space of a dataset can be observed with a simple function call below. To create it, the training features + labels and testing features + labels are passed in.

```Python
# Plot feature space 
fe.visualize_feature_space(data_set['training_features'], projection="PCA", classes=train_meta['classes'], test_feature_dic=test_features, t_classes=test_meta['classes'])
```
