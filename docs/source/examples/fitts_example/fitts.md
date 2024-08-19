[View Source Code](https://github.com/libemg/LibEMG_Isofitts_Showcase)

<style>
    .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
</style>

For EMG-based control systems, it has been shown that the offline performance of a system (i.e., classification accuracy) does not necessarily correlate to online usability. In this example, we introduce an Iso Fitts test for assessing the online performance of continuous EMG-based control systems. While this test evaluates systems with 2DOFs that leverage continuous constant control, it could be extended to more complicated systems such as proportional control with discrete inputs.

# Methods 
This example acts as a mini experiment that you can try out on yourself or a friend where the offline and online performance of four popular classifiers (**LDA, SVM, RF,** and **KNN (k=5**)) are compared. 

The steps of this 'mini experiment' are as follows:
1. **Accumulate 5 repetitions of five contractions (no movement, flexion, extension, hand open, and hand closed).** These classes correspond to movement in the isofitts task (do nothing,  and move left, right, up, and down).
    <div>
        <img src="https://github.com/libemg/LibEMG_Isofitts_Showcase/blob/main/docs/menu.PNG?raw=true" width="32%" display="inline-block" float="left"/>
        <img src="https://github.com/libemg/LibEMG_Snake_Showcase/blob/main/docs/training_screen1.PNG?raw=true" width="32%" float="left"/>
        <img src="https://github.com/libemg/LibEMG_Snake_Showcase/blob/main/docs/training_screen2.PNG?raw=true" width="32%" float="left"/>
    </div>
2. **Train and evaluate four classifiers in an offline setting (LDA, SVM, KNN (k=5), and RF).** For this step, the first three reps are used for training and the last two for testing. 
3. **Perform an Iso Fitts test to evaluate the online usability of each classifier trained in step 2.** These fitts law tests are useful for computing throughput, overshoots, and efficiency. Ultimately, these metrics provide an indication of the online usability of a model. The Iso Fitts test is useful for myoelectric control systems as it requires changes in degrees of freedom to complete sucessfully.
   
    <img src="https://github.com/libemg/LibEMG_Isofitts_Showcase/blob/main/docs/isofitts.PNG?raw=true" class="center"/>

**Note:** We have made this example to work with the `Myo Armband`. However, it can easily be used for any hardware by simply switching the `streamer`, `WINDOW_SIZE`, and `INCREMENT`.

# Menu
```Python
from libemg.streamers import myo_streamer
from libemg.gui import GUI
from libemg.data_handler import OnlineDataHandler, OfflineDataHandler, RegexFilter
from libemg.utils import make_regex
from libemg.feature_extractor import FeatureExtractor
from libemg.emg_predictor import OnlineEMGClassifier, EMGClassifier
```
Similarly to previous examples, we decided to create a simple menu to (1) leverage the training module and (2) enable the use of different classifiers. To do this, we have included two buttons in `menu.py`. When the "accumulate training data button" is clicked, we leverage the training UI module. For this example, we want five reps (3 training - 2 testing), and we point it to the "classes" folder as it contains images for each class.

```Python
def launch_training(self):
    self.window.destroy()
    training_ui = GUI(self.odh, width=700, height=700, gesture_height=300, gesture_width=300)
    training_ui.download_gestures([1,2,3,4,5], "images/")
    training_ui.start_gui()
```

The next button option involves starting the Iso Fitts task. This occurs after the training data has been recorded. Note that in this step we create the online classifier and start the Fitts law test. We opted for 8 circles, but this can be varied easily with the constructor.

```Python
def start_test(self):
    self.window.destroy()
    self.set_up_classifier()
    FittsLawTest(num_trials=8, num_circles=8, savefile=self.model_str.get() + ".pkl").run()
    # Its important to stop the classifier after the game has ended
    # Otherwise it will continuously run in a seperate process
    self.classifier.stop_running()
    self.initialize_ui()
```

Now, let's break this piece of code up. First, let's explore the `self.set_up_classifier()` function call. This step involves parsing the offline training data using the `OfflineDataHandler`. The file format for this example is R_<#>_C_<#>.csv. So to extract the reps the left bound is `R_` and the right bound is `_C_`. Similarly, to extract the classes, the left bound is `C_` and the right bound is `.csv`. Additionally, there are three training reps and five classes. Once we extract all this information, we create the `OfflineDataHandler` and extract the `train_windows` and `train_metadata` variables.

```Python
# Step 1: Parse offline training data
dataset_folder = 'data/'
classes_values = ["0","1","2","3","4"]
reps_values = ["0", "1", "2"]
regex_filters = [
    RegexFilter(left_bound = "_C_", right_bound=".csv", values = classes_values, description='classes'),
    RegexFilter(left_bound = "R_", right_bound="_C_", values = reps_values, description='reps')
]
odh = OfflineDataHandler()
odh.get_data(folder_location=dataset_folder, regex_filters=regex_filters, delimiter=",")
train_windows, train_metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
```

The next step involves extracting features from the training data. To do this we leverage the   `FeatureExtractor` module. In this example, we use the `Low Sampling 4 (LS4)` feature set as it is a robust group for low sampling rate devices such as the Myo. 

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
data_set['training_labels'] = train_metadata['classes']
```

Finally, we create the `EMGClassifier` and the `OnlineEMGClassifier` using the default options. Notice that when creating the classifier, we pass in the text from the menu text field. This enables the user to pass in `LDA`, `SVM`, etc. with ease. Once the classifier is created, the `.run()` function is called and predictions begin.

```Python
# Step 4: Create the EMG Classifier
o_classifier = EMGClassifier(self.model_str.get())
o_classifier.fit(feature_dictionary=data_set)

# Step 5: Create online EMG classifier and start classifying.
self.classifier = OnlineEMGClassifier(o_classifier, WINDOW_SIZE, WINDOW_INCREMENT, self.odh, feature_list)
self.classifier.run(block=False) # block set to false so it will run in a seperate process.
```

# Fitts Test 
To create the Isofitts test, we leveraged `pygame`. The code for this module can be found in `isofitts.py`. The cursor moves based on the `OnlineEMGClassifier's` predictions:

```Python
self.current_direction = [0,0]
data, _ = self.sock.recvfrom(1024)
data = str(data.decode("utf-8"))
if data:
    input_class = float(data.split(' ')[0])
    # 0 = Hand Closed = down
    if input_class == 0:
        self.current_direction[1] += self.VEL
    # 1 = Hand Open
    elif input_class == 1:
        self.current_direction[1] -= self.VEL
    # 3 = Extension 
    elif input_class == 3:
        self.current_direction[0] += self.VEL
    # 4 = Flexion
    elif input_class == 4:
        self.current_direction[0] -= self.VEL
```

To increase the speed of the cursor we could do one of two things: (1) increase the velocity of the cursor (i.e., how many pixels it moves for each prediction), or (2) decrease the increment so that more predictions are made in the same amount of time.

# Data Analysis
After accumulating data from the experiment, we need a way to analyze the data. In  `analyze_data.py`, we added the capability to evaluate each model's offline and online performance. 

To evaluate each model's offline performance, we took a similar approach to set up the online classifier. However, in this case, we have to split up the data into training and testing. To do this, we first extract each of the five reps of data. We will split this into training and testing in a little bit.
```Python
dataset_folder = 'data'
classes_values = ["0","1","2","3","4"]
reps_values = ["0","1","2","3","4"]
regex_filters = [
    RegexFilter(left_bound = "_C_", right_bound=".csv", values = classes_values, description='classes'),
    RegexFilter(left_bound = "R_", right_bound="_C_", values = reps_values, description='reps')
]
odh = OfflineDataHandler()
odh.get_data(folder_location=dataset_folder, filename_dic = dic, delimiter=",")
```
Using the `isolate_data` function, we can split the data into training and testing. In this specific case, we are splitting on the "reps" keyword and we want values with index 0-2 for training and 3-4 for testing. After isolating the data, we extract windows and associated metadata for both training and testing sets. 

```Python
train_odh = odh.isolate_data(key="reps", values=[0,1,2])
train_windows, train_metadata = train_odh.parse_windows(WINDOW_SIZE,WINDOW_INCREMENT)
test_odh = odh.isolate_data(key="reps", values=[3,4])
test_windows, test_metadata = test_odh.parse_windows(WINDOW_SIZE,WINDOW_INCREMENT)
```

Next, we create a dataset dictionary consisting of testing and training features and labels. This dictionary is passed into an `OfflineDataHandler.`
```Python
data_set = {}
data_set['testing_features'] = fe.extract_feature_group('HTD', test_windows)
data_set['training_features'] = fe.extract_feature_group('HTD', train_windows)
data_set['testing_labels'] = test_metadata['classes']
data_set['training_labels'] = train_metadata['classes']
```

Finally, to extract the offline performance of each model, we leverage the `OfflineMetrics` module. We do this in a loop to easily evaluate a number of classifiers. We append the metrics to a dictionary for future use.
```Python
om = OfflineMetrics()
metrics = ['CA', 'AER', 'INS', 'CONF_MAT']
# Normal Case - Test all different classifiers
for model in ['LDA', 'SVM', 'KNN', 'RF']:
    classifier = EMGClassifier()
    classifier.fit(model, data_set.copy())
    preds, probs = classifier.run(data_set['testing_features'], data_set['testing_labels'])
    out_metrics = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, 2)
    offline_metrics['classifier'].append(model)
    offline_metrics['metrics'].append(out_metrics)
return offline_metric
```

# Results
There are clear discrepancies between offline and online metrics. For example, RF outperforms LDA in the offline analysis, but it is clear in the online test that it is much worse. This highlights the need to evaluate EMG-based control systems in online settings with user-in-the-loop feedback.

**Visual Output:**
<img src="https://github.com/libemg/LibEMG_Isofitts_Showcase/blob/main/docs/perf_metrics.PNG?raw=true"/>
