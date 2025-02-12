[View Source Code](https://github.com/libemg/LibEMG_Isofitts_Showcase)

<style>
    .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
</style>

For EMG-based control systems, it has been shown that the offline performance of a system (i.e., classification accuracy, mean absolute error) does not necessarily correlate to online usability. In this example, we introduce an Iso Fitts test for assessing the online performance of continuous EMG-based control systems.
Different types of models, such as regressors and classifiers, cannot be easily compared offline since different metrics are calculated. Online tests allow us to compare these distinct model types and assess a model's ability to perform a task with a user in the loop.

# Methods 
This example acts as a mini experiment that you can try out on yourself or a friend where the offline and online performance of four popular classifiers (**LDA, SVM, RF,** and **KNN (k=5**)) and two regressors (**LR and SVM**) are compared.

The steps of this 'mini experiment' are as follows:
1. **Accumulate 3 repetitions of five contractions (no movement, flexion, extension, hand open, and hand closed).** These classes correspond to movement in the isofitts task (do nothing,  and move left, right, up, and down).

<table>
  <tr>
    <td><img src="https://github.com/libemg/LibEMG_Isofitts_Showcase/blob/main/docs/menu.PNG?raw=true" alt="Image 1" width="1200"/></td>
    <td><img src="https://github.com/libemg/LibEMG_Isofitts_Showcase/blob/main/docs/training_screen1.png?raw=true" alt="Image 2" width="1200"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/libemg/LibEMG_Isofitts_Showcase/blob/main/docs/training_screen2.png?raw=true" alt="Image 3" width="1200"/></td>
    <td><img src="https://github.com/libemg/LibEMG_Isofitts_Showcase/blob/main/docs/training_screen3.png?raw=true" alt="Image 4" width="1200"/></td>
  </tr>
</table>

2. **Train and evaluate four classifiers in an offline setting (LDA, SVM, KNN (k=5), and RF).** For this step, the first 2 reps are used for training and the last for testing. 
3. **Perform an Iso Fitts test to evaluate the online usability of each classifier trained in step 2.** These fitts law tests are useful for computing throughput, overshoots, and efficiency. Ultimately, these metrics provide an indication of the online usability of a model. The Iso Fitts test is useful for myoelectric control systems as it requires changes in degrees of freedom to complete sucessfully.
    <img src="https://github.com/libemg/LibEMG_Isofitts_Showcase/blob/main/docs/isofitts.PNG?raw=true" class="center"/>
4. **Repeat steps 1-3 using regressors instead of classifiers.** Select 'regression' from the radio buttons and redo data collection. You will now be shown a video of a point moving through a cartesian plane, which indicates the position along each DOF. Follow the point in real-time to provide the regressor with continuously-labelled training data (as opposed to classes in classification). This video will be repeated 3 times (i.e., 3 repetitions). Note that you can now perform simultaneous contractions (i.e., move the cursor along the diagonal) when using a regressor.

**Note:** We have made this example to work with the `Myo Armband`. However, it can easily be used for any hardware by simply switching the `streamer`, `WINDOW_SIZE`, and `WINDOW_INCREMENT`.

# Menu
```Python
from libemg.streamers import myo_streamer
from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter, FilePackager
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import OnlineEMGClassifier, EMGClassifier, EMGRegressor, OnlineEMGRegressor
from libemg.environments.fitts import ISOFitts, FittsConfig
from libemg.environments.controllers import ClassifierController, RegressorController
from libemg.animator import ScatterPlotAnimator
```
Similarly to previous examples, we decided to create a simple menu to (1) leverage the training module and (2) enable the use of different models. To do this, we have included two buttons in `menu.py`. When the "accumulate training data button" is clicked, we leverage the training UI module. For this example, we want 3 reps (2 training - 1 testing), and we point it to the "images" folder as it contains images for each class. To instead evaluate a regressor, simply select the 'Regression' radio button and collect the required data. Note that launching training for regression will create a `collection.mp4` file using the `Animator` class.

```Python
def launch_training(self):
    self.window.destroy()
    if self.regression_selected():
        args = {'media_folder': 'animation/', 'data_folder': Path('data', 'regression').absolute().as_posix(), 'rep_time': 50}
    else:
        args = {'media_folder': 'images/', 'data_folder': Path('data', 'classification').absolute().as_posix()}
    training_ui = GUI(self.odh, args=args, width=700, height=700, gesture_height=300, gesture_width=300)
    training_ui.download_gestures([1,2,3,4,5], "images/")
    self.create_animation()
    training_ui.start_gui()
    self.initialize_ui()

def create_animation(self):
    output_filepath = Path('animation', 'collection.mp4').absolute()
    if not self.regression_selected() or output_filepath.exists():
        return

    print('Creating regression training animation...')
    period = 2  # period of sinusoid (seconds)
    cycles = 10
    rest_time = 5 # (seconds)
    fps = 24

    coordinates = []
    total_duration = int(cycles * period + rest_time)
    t = np.linspace(0, total_duration - rest_time, fps * (total_duration - rest_time))
    coordinates.append(np.sin(2 * np.pi * (1 / period) * t))    # add sinusoids
    coordinates.append(np.zeros(fps * rest_time))   # add rest time

    # Convert into 2D (N x M) array with isolated sinusoids per DOF
    coordinates = np.expand_dims(np.concatenate(coordinates, axis=0), axis=1)
    dof1 = np.hstack((coordinates, np.zeros_like(coordinates)))
    dof2 = np.hstack((np.zeros_like(coordinates), coordinates))
    coordinates = np.vstack((dof1, dof2))
    
    axis_images = {
        'N': PILImage.open(Path('images', 'Hand_Open.png')),
        'S': PILImage.open(Path('images', 'Hand_Close.png')),
        'E': PILImage.open(Path('images', 'Wrist_Extension.png')),
        'W': PILImage.open(Path('images', 'Wrist_Flexion.png'))
    }
    animator = ScatterPlotAnimator(output_filepath=output_filepath.as_posix(), show_direction=True, show_countdown=True, axis_images=axis_images)
    animator.save_plot_video(coordinates, title='Regression Training', save_coordinates=True, verbose=True)
```

The next button option involves starting the Iso Fitts task. This occurs after the training data has been recorded. We opted for 8 circles in this task, but this can be varied easily with the constructor. Note that in this step we create the online model and start the Fitts law test, so please ensure that you have changed the text box to select the desired model type. If you are performing classification, recommended options are 'LDA', 'SVM', 'KNN', and 'RF'. If you are performing regression, recommended options are 'LR' and 'RF'. To see a full list of available options for classifiers and regressors in `LibEMG`, check out the `EMGClassifier` and `EMGRegressor` in the [source code](https://github.com/LibEMG/libemg).

```Python
def start_test(self):
    self.window.destroy()
    self.set_up_model()
    if self.regression_selected():
        controller = RegressorController()
        save_file = Path('results', self.model_str.get() + '_reg' + ".pkl").absolute().as_posix()
    else:
        controller = ClassifierController(output_format=self.model.output_format, num_classes=5)
        save_file = Path('results', self.model_str.get() + '_clf' + ".pkl").absolute().as_posix()
    config = FittsConfig(num_trials=16, save_file=save_file)
    ISOFitts(controller, config).run()
    # Its important to stop the model after the game has ended
    # Otherwise it will continuously run in a seperate process
    self.model.stop_running()
    self.initialize_ui()
```

Now, let's break this piece of code up. First, let's explore the `self.set_up_model()` function call. This step involves parsing the offline training data using the `OfflineDataHandler`. The file format for this example is R_<#>_C_<#>.csv. So to extract the reps the left bound is `R_` and the right bound is `_C_`. Similarly, to extract the classes, the left bound is `C_` and the right bound is `.csv`. Additionally, there are three training reps and five classes. For regression, the labels are extracted from a separate text file instead of from the filename. The `collection.txt` file is used to create the labels for each data file via a `FilePackager`. Once we extract all this information, we create the `OfflineDataHandler` and extract the `train_windows` and `train_metadata` variables.

```Python
# Step 1: Parse offline training data
if self.regression_selected():
    regex_filters = [
        RegexFilter(left_bound='regression/C_0_R_', right_bound='_emg.csv', values=['0', '1', '2'], description='reps')
    ]
    metadata_fetchers = [
        FilePackager(RegexFilter(left_bound='animation/', right_bound='.txt', values=['collection'], description='labels'), package_function=lambda x, y: True)
    ]
    labels_key = 'labels'
    metadata_operations = {'labels': 'last_sample'}
else:
    regex_filters = [
        RegexFilter(left_bound = "classification/C_", right_bound="_R", values = ["0","1","2","3","4"], description='classes'),
        RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = ["0", "1", "2"], description='reps'),
    ]
    metadata_fetchers = None
    labels_key = 'classes'
    metadata_operations = None

odh = OfflineDataHandler()
odh.get_data('./', regex_filters, metadata_fetchers=metadata_fetchers, delimiter=",")
train_windows, train_metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT, metadata_operations=metadata_operations)
```

The next step involves extracting features from the training data. To do this we leverage the `FeatureExtractor` module. In this example, we use the `Low Sampling 4 (LS4)` feature set as it is a robust group for low sampling rate devices such as the Myo.

```Python
# Step 2: Extract features from offline data
fe = FeatureExtractor()
feature_list = fe.get_feature_groups()['LS4']
training_features = fe.extract_features(feature_list, train_windows)
```

We then split the training features and labels into a dataset dictionary for the `OnlineEMGClassifier` module. 

```Python
# Step 3: Dataset creation
data_set = {}
data_set['training_features'] = training_features
data_set['training_labels'] = train_metadata[labels_key]
```

Finally, we create the offline and online models using the default options. Notice that when creating the model, we pass in the text from the menu text field. This enables the user to pass in `LDA`, `SVM`, etc. with ease. Once the model is created, the `.run()` function is called and predictions begin.

```Python
# Step 4: Create the EMG model
model = self.model_str.get()
if self.regression_selected():
    # Regression
    emg_model = EMGRegressor(model=model)
    emg_model.fit(feature_dictionary=data_set)
    self.model = OnlineEMGRegressor(emg_model, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list)
else:
    # Classification
    emg_model = EMGClassifier(model=model)
    emg_model.fit(feature_dictionary=data_set)
    emg_model.add_velocity(train_windows, train_metadata[labels_key])
    self.model = OnlineEMGClassifier(emg_model, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list output_format='probabilities')

# Step 5: Create online EMG model and start predicting.
self.model.run(block=False) # block set to false so it will run in a seperate process.
```

# Fitts Test 
To create the Isofitts test, we leveraged `pygame`. The code for this module can be found in `libemg.environments.isofitts.py`.

To increase the speed of the cursor we could do one of two things: (1) increase the velocity of the cursor (i.e., how many pixels it moves for each prediction), or (2) decrease the increment so that more predictions are made in the same amount of time. Parameters like this can be modified by passing arguments to the `IsoFitts` constructor.

# Data Analysis
After accumulating data from the experiment, we need a way to analyze the data. In  `analyze_data.py`, we added the capability to evaluate each model's offline and online performance.

To evaluate each model's offline performance, we took a similar approach to set up the online model. However, in this case, we have to split up the data into training and testing. To do this, we first extract each of the 3 reps of data. We will split this into training and testing in a little bit.
```Python
regex_filters = [
    RegexFilter(left_bound = "classification/C_", right_bound="_R", values = ["0","1","2","3","4"], description='classes'),
    RegexFilter(left_bound = "R_", right_bound="_emg.csv", values = ["0", "1", "2"], description='reps'),
]

clf_odh = OfflineDataHandler()
clf_odh.get_data('data/', regex_filters, delimiter=",")

regex_filters = [
    RegexFilter(left_bound='data/regression/C_0_R_', right_bound='_emg.csv', values=['0', '1', '2'], description='reps')
]
metadata_fetchers = [
    FilePackager(RegexFilter(left_bound='animation/', right_bound='.txt', values=['collection'], description='labels'), package_function=lambda x, y: True)
]
reg_odh = OfflineDataHandler()
reg_odh.get_data('./', regex_filters, metadata_fetchers=metadata_fetchers, delimiter=',')
```

Using the `isolate_data` function, we can split the data into training and testing. In this specific case, we are splitting on the "reps" keyword and we want values with index 0-1 for training and 2 for testing. After isolating the data, we extract windows and associated metadata for both training and testing sets.

```Python
train_odh = odh.isolate_data(key="reps", values=[0,1])
train_windows, train_metadata = train_odh.parse_windows(WINDOW_SIZE,WINDOW_INCREMENT, metadata_operations=metadata_operations)
test_odh = odh.isolate_data(key="reps", values=[2])
test_windows, test_metadata = test_odh.parse_windows(WINDOW_SIZE,WINDOW_INCREMENT, metadata_operations=metadata_operations)
```

Next, we create a dataset dictionary consisting of testing and training features and labels. This dictionary is passed into an `OfflineDataHandler`.
```Python
data_set = {}
data_set['testing_features'] = fe.extract_feature_group('HTD', test_windows)
data_set['training_features'] = fe.extract_feature_group('HTD', train_windows)
data_set['testing_labels'] = test_metadata[labels_key]
data_set['training_labels'] = train_metadata[labels_key]
```

Finally, to extract the offline performance of each model, we leverage the `OfflineMetrics` module. We do this in a loop to easily evaluate a number of models. We append the metrics to a dictionary for future use.
```Python
om = OfflineMetrics()
# Normal Case - Test all different models
for model in models:
    if is_regression:
        model = EMGRegressor(model)
        model.fit(data_set.copy())
        preds = model.run(data_set['testing_features'])
    else:
        model = EMGClassifier(model)
        model.fit(data_set.copy())
        preds, _ = model.run(data_set['testing_features'])
    out_metrics = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, 2)
    offline_metrics['model'].append(model)
    offline_metrics['metrics'].append(out_metrics)
return offline_metrics
```

# Results
There are clear discrepancies between offline and online metrics. For example, RF outperforms LDA in the offline classification analysis, but it is clear in the online test that it is much worse. Similarly, RF outperforms LR in the offline regression analysis, but the usability metrics again suggest that LR outperforms RF during an online task. This highlights the need to evaluate EMG-based control systems in online settings with user-in-the-loop feedback.

These results also show that regressors had worse usability metrics than classifiers despite enabling simultaneous motions. The high number of overshoots indicate that this is likely due to the fact that the model stuggled to stay at rest without drifting, increasing the time each trial took. This example could be expanded by adding things like a threshold to the regressors (see `EMGRegressor.add_deadband`), which may improve regressor performance.

**Visual Output:**
<img src="https://github.com/libemg/LibEMG_Isofitts_Showcase/blob/main/docs/perf_metrics.PNG?raw=true"/>
