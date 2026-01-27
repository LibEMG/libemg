[View Source Code](https://github.com/LibEMG/LibEMG_OfflineRegression_Showcase)

<img src="https://github.com/LibEMG/LibEMG_OfflineRegression_Showcase/blob/main/docs/results.png?raw=true"/>

This simple offline example showcases some of the offline capabilities for regression analysis. In this example, we will load in the OneSubjectEMaGerDataset and assess the performance of multiple regressors. All code can be found in `main.py`.

## Step 1: Importing LibEMG

The very first step involves importing the modules needed. In general, each of LibEMG's modules has its own import. Make sure that you have successfully installed libemg through pip.

```Python
import numpy as np
import matplotlib.pyplot as plt
from libemg.offline_metrics import OfflineMetrics
from libemg.datasets import OneSubjectEMaGerDataset
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import EMGRegressor
```

## Step 2: Setting up Constants

Preprocessing parameters, such as window size, window increment, and the feature set must be decided before EMG data can be prepared for estimation. LibEMG defines window and increment sizes as the number of samples. In this case, the dataset was recorded from the EMaGer cuff, which samples at 1 kHz, so a window of 150 samples corresponds to 150ms.

The window increment, window size, and feature set default to 40, 150, and 'HTD', respecively. These variables can be customized in this script using the provided CLI. Use `python main.py -h` for an explanation of the CLI. Example usage is also provided below:

```Bash
python main.py --window_size 200 --window_increment 50 --feature_set MSWT
```

# Step 3: Loading in Dataset

This example uses the `OneSubjectEMaGerDataset`. Instantiating the `Dataset` will automatically download the data into the specified directory, and calling the `prepare_data()` method will load EMG data and metadata (e.g., reps, movements, labels) into an `OfflineDataHandler`. This dataset consists of 5 repetitions, so we use 4 for training data and 1 for testing data. After splitting our data into training and test splits, we perform windowing on the raw EMG data. By default, the metadata assigned to each window will be based on the mode of that window. Since we are analyzing regression data, we pass in a function that tells the `OfflineDataHandler` to grab the label from the last sample in the window instead of taking the mode of the window. We can specify how we want to handle windowing of each type of metadata by passing in a `metadata_operations` dictionary.

```Python
# Load data
data = OneSubjectEMaGerDataset().prepare_data()

# Split into train/test reps
train_odh = data['Train']
test_odh = data['Test']

# Extract windows
metadata_operations = {'labels': lambda x: x[-1]}   # grab label of last sample in window
train_windows, train_metadata = train_odh.parse_windows(args.window_size, args.window_increment, metadata_operations=metadata_operations)
test_windows, test_metadata = test_odh.parse_windows(args.window_size, args.window_increment, metadata_operations=metadata_operations)
```

# Step 4: Feature Extraction

We then extract features using the `FeatureExtractor` for our training and test data. The `fit()` method expects a dictionary with the keys `training_features` and `training_labels`, so we create one and pass in our extracted features and training labels.

```Python
training_features = fe.extract_feature_group(args.feature_set, train_windows, array=True),
training_labels = train_metadata['labels']
test_features = fe.extract_feature_group(args.feature_set, test_windows, array=True)
test_labels = test_metadata['labels']

training_set = {
    'training_features': training_features,
    'training_labels': training_labels
}
```

# Step 5: Regression

`LibEMG` allows you to pass in custom models, but you can also pass in a string that will create a model for you. In this example, we compare a linear regressor to a gradient boosting regressor. We iterate through a list of the models we want to observe, fit the model to the training data, and calculate metrics based on predictions on the test data. We then store these metrics for plotting later.

```Python
results = {metric: [] for metric in ['R2', 'NRMSE', 'MAE']}
for model in models:
    reg = EMGRegressor(model)

    # Fit and run model
    print(f"Fitting {model}...")
    reg.fit(training_set.copy())
    predictions = reg.run(test_features)

    metrics = om.extract_offline_metrics(results.keys(), test_labels, predictions)
    for metric in metrics:
        results[metric].append(metrics[metric].mean())
```

# Step 6: Visualization

Finally, we visualize our results. We first plot the decision stream for each model. After each model is fitted, we plot the offline metrics for each type of model.

```Python
# Note: this will block the main thread once the plot is shown. Close the plot to continue execution.
reg.visualize(test_labels, predictions)

fig, axs = plt.subplots(nrows=len(results), layout='constrained', figsize=(8, 8), sharex=True)
for metric, ax in zip(results.keys(), axs):
    ax.bar(models, np.array(results[metric]) * 100)
    ax.set_ylabel(f"{metric} (%)")

fig.suptitle('Metrics Summary')
plt.show()
```
