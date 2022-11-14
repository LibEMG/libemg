<style>
    .center {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 50%;
    }
</style>
[View Source Code](https://github.com/eeddy/Isofitts)

For EMG-based control systems, it has been shown that the offline performance of a system does not necessarily correlate to online usability. This example tests this idea through a simple experiment that you can try out on yourself or a friend. To do this we evaluate and compare the offline and online performance of four popular classifiers (**LDA, SVM, NB,** and **KNN (k=5**)). 

# Methods 
The steps of this 'mini experiment' are as follows:
1. **Accumulate 5 repetitions of five contractions (no movement, flexion, extension, hand open, and hand closed).** These classes correspond to movement in the isofitts task (left, right, up, and down).
    <div>
        <img src="https://github.com/eeddy/Isofitts/blob/main/docs/menu.PNG?raw=true" width="32%" display="inline-block" float="left"/>
        <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/training_screen1.PNG?raw=true" width="32%" float="left"/>
        <img src="https://github.com/eeddy/Snake-Demo/blob/main/docs/training_screen2.PNG?raw=true" width="32%" float="left"/>
    </div>
2. **Train and evaluate four classifiers in an offline setting (LDA, SVM, KNN (k=5), and NB).** For this step, the first three reps are used for training and the last two for testing. 
3. **Perform an isofitts test to evaluate the online usability of each classifier trained in step 2.** These fitts law tests are useful for computing throughput, overshoots, and efficiency. Ultimately, these metrics provide an indication of the online usability of a model. 
   
    <img src="https://github.com/eeddy/Isofitts/blob/main/docs/isofitts.PNG?raw=true" class="center"/>

**Note:** We have made this example to work with the `Myo Armband`. However, it can easily be used for any hardware by simply switching the `num_channels`, `WINDOW_SIZE`, and `INCREMENT` variables.

# Menu
```Python
from unb_emg_toolbox.training_ui import TrainingUI
from unb_emg_toolbox.data_handler import OnlineDataHandler, OfflineDataHandler
from unb_emg_toolbox.utils import make_regex
from unb_emg_toolbox.feature_extractor import FeatureExtractor
from unb_emg_toolbox.emg_classifier import OnlineEMGClassifier
```
Similarly to previous examples, we decided to create a simple menu to (1) leverage the training module and (2) enable the use of different classifiers. To do this, we have included two buttons in `menu.py`. When the "accumulate training data button" is clicked, we leverage the training UI module. For this example, we want five reps (3 training - 2 testing), and we point it to the "classes" folder as it contains images for each class.

```Python
def launch_training(self):
    self.window.destroy()
    # Launch training ui
    TrainingUI(num_reps=5, rep_time=5, rep_folder="classes/", output_folder="data/", data_handler=self.odh)
    self.initialize_ui()
```

The next button option involves starting the Isofitts task. This occurs after the training data has been recorded. Note that in this step we create the online classifier and start the Fitts law test. 

```Python
def start_test(self):
    self.window.destroy()
    self.set_up_classifier()
    FittsLawTest(num_trials=5, savefile=self.model_str.get() + ".pkl").run()
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
classes_regex = make_regex(left_bound = "_C_", right_bound=".csv", values = classes_values)
reps_values = ["0", "1", "2"]
reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
dic = {
    "reps": reps_values,
    "reps_regex": reps_regex,
    "classes": classes_values,
    "classes_regex": classes_regex
}

odh = OfflineDataHandler()
odh.get_data(folder_location=dataset_folder, filename_dic=dic, delimiter=",")
train_windows, train_metadata = odh.parse_windows(WINDOW_SIZE, WINDOW_INCREMENT)
```

The next step involves extracting features from the training data. To do this we leverage the   `FeatureExtractor` module. In this example, we use the `Hudgin's Time Domain (HTD)` feature set as they are quite popular. 

```Python
# Step 2: Extract features from offline data
fe = FeatureExtractor(num_channels=8)
feature_list = fe.get_feature_groups()['HTD']
training_features = fe.extract_features(feature_list, train_windows)
```

We then split the training features and labels into a dataset dictionary for the `OnlineEMGClassifier` module. 

```Python
# Step 3: Dataset creation
data_set = {}
data_set['training_features'] = training_features
data_set['training_labels'] = train_metadata['classes']
```

Finally, we create the `OnlineEMGClassifier` using the default options. Notice that when creating the classifier, we pass in the text from the menu text field. This enables the user to pass in `LDA`, `SVM`, etc. with ease. Once the classifier is created, the `.run()` function is called and predictions begin.

```Python
 # Step 4: Create online EMG classifier and start classifying.
self.classifier = OnlineEMGClassifier(model=self.model_str.get(), data_set=data_set, num_channels=8, window_size=WINDOW_SIZE, window_increment=WINDOW_INCREMENT, 
        online_data_handler=self.odh, features=feature_list)
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

# Data Analysis
After accumulating data from the experiment, we need a way to analyze the data. In  `analyze_data.py`, we added the capability to evaluate each model's offline and online performance. 

To evaluate each model's offline performance, we took a similar approach to set up the online classifier. However, in this case, we have to split up the data into training and testing. To do this, we first extract each of the five reps of data. We will split this into training and testing in a little bit.
```Python
dataset_folder = 'data'
classes_values = ["0","1","2","3","4"]
classes_regex = make_regex(left_bound = "_C_", right_bound=".csv", values = classes_values)
reps_values = ["0","1","2","3","4"]
reps_regex = make_regex(left_bound = "R_", right_bound="_C_", values = reps_values)
dic = {
    "reps": reps_values,
    "reps_regex": reps_regex,
    "classes": classes_values,
    "classes_regex": classes_regex
}
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
metrics = ['CA', 'AER', 'INS', 'RECALL', 'PREC', 'F1']
# Normal Case - Test all different classifiers
for model in ['LDA', 'SVM', 'KNN', 'NB']:
    classifier = EMGClassifier(model, data_set.copy())
    preds = classifier.run()
    out_metrics = om.extract_offline_metrics(metrics, data_set['testing_labels'], preds, 2)
    offline_metrics['classifier'].append(model)
    offline_metrics['metrics'].append(out_metrics)
return offline_metrics
```


# Results
Looking at the results, there are clear discrepancies between offline and online metrics. For example, the LDA classifier is the most performant during offline analysis, however, this does not necessarily translate in this specific case to online usability. 

**Visual Output:**
<img src="https://github.com/eeddy/Isofitts/blob/main/docs/perf_metrics.PNG?raw=true"/>

**Standard Output:**
```Python
{'classifier': ['LDA', 'SVM', 'KNN', 'NB'], 'metrics': [{'CA': 0.9848024316109423, 'AER': 0.01883239171374762, 'INS': 0.016210739614994935, 'RECALL': 0.9848024316109422, 'PREC': 0.9852176594066209, 'F1': 0.9848684130766312}, {'CA': 0.975177304964539, 'AER': 0.03099304237824163, 'INS': 0.021783181357649443, 'RECALL': 0.9751773049645389, 'PREC': 0.9760464804555915, 'F1': 0.9751171560838843}, {'CA': 0.958966565349544, 'AER': 0.05100755667506296, 'INS': 0.03090172239108409, 'RECALL': 0.958966565349544, 'PREC': 0.963626568657454, 'F1': 0.9586036066370233}, {'CA': 0.9670719351570415, 'AER': 0.04082914572864327, 'INS': 0.025835866261398176, 'RECALL':
0.9670719351570415, 'PREC': 0.9691383140798603, 'F1': 0.9670068537549226}]}

{'classifier': ['LDA.pkl', 'SVM.pkl', 'KNN.pkl', 'NB.pkl'], 'metrics': [{'overshoots': 17, 'throughput': 0.2875643578931376, 'efficiency': 0.5088560433826349}, {'overshoots': 24, 'throughput': 0.3001490731763605, 'efficiency': 0.592053447600828}, {'overshoots': 13, 'throughput': 0.2979748441587429, 'efficiency': 0.5634924034816269}, {'overshoots': 16, 'throughput': 0.2526577634230205, 'efficiency': 0.4344741794266368}]}
```