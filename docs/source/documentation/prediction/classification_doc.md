# Classifiers

Below is a list of the classifiers that can be instatiated by passing in a string to the `EMGClassifier`. For other classifiers, pass in a custom model that has the `fit`, `predict`, and `predict_proba` methods.

## Linear Discriminant Analysis (LDA)

A linear classifier that uses common covariances for all classes and assumes a normal distribution.
```Python
classifier = EMGClassifier('LDA')
classifier.fit(data_set)
```
Check out the LDA docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

## K-Nearest Neighbour (KNN)

Discriminates between inputs using the K closest samples in feature space. The implemented version in the library defaults to k = 5. A commonly used classifier for EMG-based recognition.

```Python
params = {'n_neighbors': 5} # Optional
classifier = EMGClassifier('KNN')
classifier.fit(data_set, parameters=params)
```
Check out the KNN docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

## Support Vector Machines (SVM)

A hyperplane that maximizes the distance between classes is used as the boundary for recognition. A commonly used classifier for EMG-based recognition.
```Python
classifier = EMGClassifier('SVM')
classifier.fit(data_set)
```
Check out the SVM docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

## Artificial Neural Networks (MLP)

A deep learning technique that uses human-like "neurons" to model data to help discriminate between inputs. Especially for this model, we **highly** recommend you create your own.
```Python
classifier = EMGClassifier('MLP')
classifier.fit(data_set)
```
Check out the MLP docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

## Random Forest (RF)

Uses a combination of decision trees to discriminate between inputs.
```Python
classifier = EMGClassifier('RF')
classifier.fit(data_set)
```
Check out the RF docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Quadratic Discriminant Analysis (QDA)

A quadratic classifier that uses class-specific covariances and assumes normally distributed classes.
```Python
classifier = EMGClassifier('QDA')
classifier.fit(data_set)
```
Check out the QDA docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)

## Gaussian Naive Bayes (NB)

Assumes independence of all input features and normally distributed classes.
```Python
classifier = EMGClassifier('NB')
classifier.fit(data_set)
```
Check out the NB docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

<!-- ### Gradient Boosting (GB)
```Python
classifier = EMGClassifier('GB', data_set)
classifier.fit('GB', data_set)
```
Check out the GB docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) -->
