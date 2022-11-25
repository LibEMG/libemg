# Classifiers
The classifiers that we have implemented all leverage the sklearn package. For most cases, the "base" classifiers use the default options, meaning that the pre-defined models are not necessarily optimal. However, you can use the `parameters` attribute when initializing the classifiers to pass in additional sklearn parameters in a dictionary. For example, looking at the `RandomForestClassifier` docs on sklearn:

![Random Forest](random_forest.png)

We could create a classifier with any of those parameters using the `parameters` attribute. For example:
```Python
parameters = {
    'n_estimators': 99,
    'max_depth': 20,
    'random_state': 5,
    'max_leaf_nodes': 10
}
classifier = EMGClassifier('RF', data_set, parameters=parameters)
```

Please reference the sklearn docs for parameter options for each classifier. 

Additionally, we can also create our own classifiers. Any custom classifier should be modeled after the sklearn classifiers and must have the `fit`, `predict`, and `predict_proba` functions to work correctly. 

```Python
from sklearn.ensemble import RandomForestClassifier
from unb_emg_toolbox import EMGClassifier

rf_custom_classifier = RandomForestClassifier(max_depth=5, random_state=0)
classifier = EMGClassifier(rf_custom_classifier, data_set)
classifier.run()
```

## Linear Discriminant Analysis (LDA)
A statistical linear boundary is used to discriminate between inputs. 
```Python
classifier = EMGClassifier('LDA', data_set)
```
Check out the LDA docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)

## K-Nearest Neighbour (KNN)
Discriminates between inputs using the K closest samples in feature space. The implemented version in the toolkit defaults to k = 5.
```Python
params = {'n_neighbors': 5} # Optional
classifier = EMGClassifier('KNN', data_set, parameters=params)
```
Check out the KNN docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

## Support Vector Machines (SVM)
A hyperplane that maximizes the distance between classes is used as the boundary for recognition.
```Python
classifier = EMGClassifier('SVM', data_set)
```
Check out the SVM docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

## Artificial Neural Networks (MLP)
A deep learning technique that uses human-like "neurons" to model data to help discriminate between inputs. Especially for this model, we **highly** recommend you create your own.
```Python
classifier = EMGClassifier('MLP', data_set)
```
Check out the MLP docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

## Random Forest (RF)
Uses a combination of decision trees to discriminate between inputs.
```Python
classifier = EMGClassifier('RF', data_set)
```
Check out the RF docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

## Quadratic Discriminant Analysis (QDA)
A statistical quadratic boundary is used to discriminate between inputs.
```Python
classifier = EMGClassifier('QDA', data_set)
```
Check out the QDA docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html)

## Gaussian Naive Bayes (NB)
Similar to LDA, but has a more simplistic model of class covariances. 
```Python
classifier = EMGClassifier('NB', data_set)
```
Check out the NB docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

## Gradient Boosting (GB)
TODO: Evan
```Python
classifier = EMGClassifier('GB', data_set)
```
Check out the GB docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)