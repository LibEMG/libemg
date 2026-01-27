Feature selection is a valuable process for optimizing pattern recognition pipelines where a large, varied representation of the input is provided (for instance a large number or features, or all principal components of the input space). Much like the motivation for feature extraction is to improve the information density from the original input signal to more meaningful features, the process of feature extraction takes many features and determines the most relevant features that provide valuable information for a particular classification task. 

In this library, we provide two methods for feature selection: metric-based selection and sequential forward selection. Metric-based selection is a single value that gets computed per feature, which is used to rank the features in terms of their effectiveness on their own. Unlike metric-based selection, where only a single round is used to assess features independent contribution, sequential foward selection is an iterative process. First, the best feature is determined in isolation based on the selected metric. The next round of the selection procedure determines the second best feature to be used in combination with the best feature determined in the last round. As this process continues, the best new feature to add in the proposed feature set is always determined in combination with the features determined in the previous rounds.

# Selection Strategies
Since EMG patterns during real-time use are often drastically different than those used to train the system, classification accuracy does not necessarily correlate to online usability. As such, there are numerous feature selection strategies for EMG-based control systems for optimizing the features selected for a particular problem. For a summary of the following feature selection techniques and suggestions, please see *A Multivariate Approach for Predicting Myoelectric Control Usability* <sup>[1]</sup>.

## Feature Extraction Setup
```python
# extract data
dataset = OneSubjectMyoDataset(save_dir='example_data', redownload=False)
odh = dataset.prepare_data(format=OfflineDataHandler)

# get the training data from the dataset
train_data = odh.isolate_data("sets",[0])
train_windows, train_metadata = train_data.parse_windows(50, 25)

# select the features for feature selection - in this case we select all of them
fe = FeatureExtractor()
feature_list = fe.get_feature_list()

# extract every feature in the toolbox on the training data
training_features = fe.extract_features(feature_list, train_windows)

# set values for feature selection
class_var = train_metadata["classes"].astype(int)
crossvalidation_var = {"var": train_metadata["reps"].astype(int)}
```

## **Classification Accuracy (CA)**
Emphasizes classification accuracy of a classifier using leave-one-trial-out cross validation.

$
\text{CA} = \frac{1}{T}\sum_{t=1}^{n}(\frac{1}{n}\sum_{i=1}^{n}\hat{y}_{i,t} == {y}_{i,t})
$

where $T$ is the total number of trials, $t$ is an individual trial, $n$ is the total number of data frames, $\hat{y}_{i,t}$ is the predicted class label for frame $i$, and $y_{i,t}$ is the true class label for frame i.

```Python
# Classification Accuracy:
metric="accuracy"
accuracy_results, accuracy_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
```

## **Active Error (AE)**
Similar to CA but focuses on active error (i.e., the error of the classifier without considering incorrect No Movmenet predictions). Also known as (1 - active classification accuracy). The active error metric assumes that class label 0 is associated with the no motion class. The easiest way to ensure this is to make the first element of the class_values key of the offlinedatahandler dictionary the no motion class.

$
\text{AE} = \frac{1}{T}\sum_{t=1}^{n}(1-(\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i,t} == {y}_{i,t}) \text{ or } (\hat{y}_{i,t} == {y}_{NM})))
$

where $T$ is the total number of trials, $t$ is an individual trial, $n$ is the total number of data frames, $\hat{y}_{i,t}$ is the predicted class label for frame $i$, $y_{i,t}$ is the true class label for frame $i$ and $y_{NM}$ is the no movement class.

```Python
# Active Error:
metric="activeerror"
aerror_results, aerror_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
```

## **Mean Semi Principle Axis (MSA)**
Quantifies the size of a training elipsoid.

$
\text{MSA} = \frac{1}{N}\sum_{j=1}^{N}((\prod_{k=1}^{D}a_{j,k})^{\frac{1}{D}})
$

where $N$ is the number of classes, $j$ represents a specific class, $D$ is the total dimensionality of the space, and $a_{k}$ is the geometric mean of each semi-principle axis.

```Python
# Mean Semi Principal Axis
metric="meansemiprincipalaxis"
msa_results, msa_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
```

## **Feature Efficiency (FE)**
A measure of the fraction of samples seperable by a particular feature.

$
\text{FE} = \frac{1}{N}\sum_{j=1}^{N}\max\limits_{i=1,...,j-1,j+1,...,N} \\ \times (\max\limits_{k=1,...,D} \frac{n(C_{i}) + n(C_{j}) - n(S_{k})}{n(C_{i}) + n(C_{j})})
$

$
S_{k} = p | p \space \epsilon \space C_{i} \space \cup \space C_{j}, \space \min(\max(f_{k}|c_{j}), \max(f_{k}|c_{i})) \ge p \ge \max(\min(f_{k}|c_{j}), \min(f_{k}|c_{i}))
$

where $N$ is the number of active classes, $j$ and $i$ are particular class labels, $D$ is the dimensionality, $n(C_{i})$ is the cardinality of the set of data points in class $i$ and $n(C_{j})$ in class $j$, $S_{k}$ is the set of points not seperable along a feature dimension $k$, $n(S_{k})$ is the cardinality of the overlap set, $f_{k}|c_{i}$ is the value of feature $f$ in dimension $k$ for class label $i$, and $p$ is the $D$ dimensional data point in class $i$ or $j$.

```Python
# Feature Efficiency:
metric="featureefficiency"
fe_results, fe_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
```

## **Repeatability Index (RI)**
Measures the repeatability of contractions/emg patterns between subsequent repetitions.

$
\text{RI} = \frac{1}{N}\sum_{j=1}^{N}\frac{1}{R}\sum_{r=1}^{R}\frac{1}{2}\sqrt{(\mu_{TR_{j}} -\mu_{r,j})'S_{TR_{j}}^-1 (\mu_{TR_{j}} - \mu_{r,j})}
$

where $N$ is the number of active classes, $R$ is the number of repetitions, $j$ is a specific class, $r$ is a specific repetition, $\mu_{TR_{j}}$ is the centroid of the class j training ellipsoid, $\mu_{r,j}$ is the centroid of a testing elipsoid of the same $j$ and rep $r$, and $S_{TR_{j}}$ is the covariance matrix of the class j training ellipsoid.

```Python
# Repeatability:
metric="repeatability"
ri_results, ri_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
```

## **Seperability Index (SI)**
A measure of the distance between different classes.

$
\text{FE} = \frac{1}{N}\sum_{j=1}^{N}\max\limits_{i=1,...,j-1,j+1,...,N} \frac{1}{2} \\ \times \sqrt{(\mu_{TR_{j}} -\mu_{TR_{i}})'S_{TR_{j}}^-1 (\mu_{TR_{j}} - \mu_{TR_{i}})}
$

where $N$ is the number of active classes, $\mu_{TR_{j}}$ is the centroid of class $j$ training ellipsoid (all reps), $\mu_{TR_{i}}$ is the centroid of the nearest training ellipsoid of a different class i, and $S_{TR_{j}}$ is the covariance matrix of class j training ellipsoid.

```Python
# Seperability:
metric="seperability"
si_results, si_fs = fs.run_selection(training_features, metric, class_var, crossvalidation_var)
```

# References
<a id="1">[1]</a> 
J. L. Nawfel, K. B. Englehart and E. J. Scheme, "A Multi-Variate Approach to Predicting Myoelectric Control Usability," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 29, pp. 1312-1327, 2021, doi: 10.1109/TNSRE.2021.3094324.