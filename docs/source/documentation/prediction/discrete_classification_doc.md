# Discrete Classifiers

Unlike continuous classifiers that output a prediction for every window of EMG data, discrete classifiers are designed for recognizing transient, isolated gestures. These classifiers operate on variable-length templates (sequences of windows) and are well-suited for detecting distinct movements like finger snaps, taps, or quick hand gestures.

Discrete classifiers expect input data in a different format than continuous classifiers:
- **Continuous classifiers**: Operate on individual windows of shape `(n_windows, n_features)`.
- **Discrete classifiers**: Operate on templates (sequences of windows) where each template has shape `(n_frames, n_features)` and can vary in length.

To prepare data for discrete classifiers, use the `discrete=True` parameter when calling `parse_windows()` on your `OfflineDataHandler`:

```Python
from libemg.data_handler import OfflineDataHandler

odh = OfflineDataHandler()
odh.get_data('./data/', regex_filters)
windows, metadata = odh.parse_windows(window_size=50, window_increment=10, discrete=True)
# windows is now a list of templates, one per file/rep
```

For feature extraction with discrete data, use the `discrete=True` parameter:

```Python
from libemg.feature_extractor import FeatureExtractor

fe = FeatureExtractor()
features = fe.extract_features(['MAV', 'ZC', 'SSC', 'WL'], windows, discrete=True, array=True)
# features is a list of arrays, one per template
```

## Majority Vote LDA (MVLDA)

A classifier that applies Linear Discriminant Analysis (LDA) to each frame within a template and uses majority voting to determine the final prediction. This approach is simple yet effective for discrete gesture recognition.

```Python
from libemg._discrete_models import MVLDA

model = MVLDA()
model.fit(train_features, train_labels)
predictions = model.predict(test_features)
probabilities = model.predict_proba(test_features)
```

## Dynamic Time Warping Classifier (DTWClassifier)

A template-matching classifier that uses Dynamic Time Warping (DTW) distance to compare test samples against stored training templates. DTW is particularly useful when gestures may vary in speed or duration, as it can align sequences with different temporal characteristics.

```Python
from libemg._discrete_models import DTWClassifier

model = DTWClassifier(n_neighbors=3)
model.fit(train_features, train_labels)
predictions = model.predict(test_features)
probabilities = model.predict_proba(test_features)
```

The `n_neighbors` parameter controls how many nearest templates are used for voting (k-nearest neighbors with DTW distance).

## Pretrained Myo Cross-User Model (MyoCrossUserPretrained)

A pretrained deep learning model for cross-user discrete gesture recognition using the Myo armband. This model uses a convolutional-recurrent architecture and recognizes 6 gestures: Nothing, Close, Flexion, Extension, Open, and Pinch.

```Python
from libemg._discrete_models import MyoCrossUserPretrained

model = MyoCrossUserPretrained()
# Model is automatically downloaded on first use

# The model provides recommended parameters for OnlineDiscreteClassifier
print(model.args)
# {'window_size': 10, 'window_increment': 5, 'null_label': 0, ...}

predictions = model.predict(test_data)
probabilities = model.predict_proba(test_data)
```

This model expects raw windowed EMG data (not extracted features) with shape `(batch_size, seq_len, n_channels, n_samples)`.

## Online Discrete Classification

For real-time discrete gesture recognition, use the `OnlineDiscreteClassifier`:

```Python
from libemg.emg_predictor import OnlineDiscreteClassifier
from libemg._discrete_models import MyoCrossUserPretrained

# Load pretrained model
model = MyoCrossUserPretrained()

# Create online classifier
classifier = OnlineDiscreteClassifier(
    odh=online_data_handler,
    model=model,
    window_size=model.args['window_size'],
    window_increment=model.args['window_increment'],
    null_label=model.args['null_label'],
    feature_list=model.args['feature_list'],  # None for raw data
    template_size=model.args['template_size'],
    min_template_size=model.args['min_template_size'],
    gesture_mapping=model.args['gesture_mapping'],
    buffer_size=model.args['buffer_size'],
    rejection_threshold=0.5,
    debug=True
)

# Start recognition loop
classifier.run()
```

## Creating Custom Discrete Classifiers

Any custom discrete classifier should implement the following methods to work with LibEMG:

- `fit(x, y)`: Train the model where `x` is a list of templates and `y` is the corresponding labels.
- `predict(x)`: Return predicted class labels for a list of templates.
- `predict_proba(x)`: Return predicted class probabilities for a list of templates.

```Python
class CustomDiscreteClassifier:
    def __init__(self):
        self.classes_ = None

    def fit(self, x, y):
        # x: list of templates (each template is an array of frames)
        # y: labels for each template
        self.classes_ = np.unique(y)
        # ... training logic

    def predict(self, x):
        # Return array of predictions
        pass

    def predict_proba(self, x):
        # Return array of shape (n_samples, n_classes)
        pass
```