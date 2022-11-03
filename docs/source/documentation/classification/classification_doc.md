# Classifiers
The classifiers that we have implemented all leverage the sklearn package. For most cases, we leveraged the default options, meaning that pre-defined models are not necessarily optimal. For more control, we suggest creating your own classifier and passing it in as the `model` parameter. Note that any custom classifier should be modeled after the sklearn classifiers and must have the `fit`, `predict`, and `predict_proba` functions. 

## Linear Discriminant Analysis (LDA)
A statistical linear boundary is used to discriminate between inputs. This is the standard for non-temporal models today.
```Python
# default
classifier = EMGClassifier('LDA', data_set)

# or passing in a custom LDA sklearn classifier 
lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()
classifier = EMGClassifier(lda, data_set)
```

## Support Vector Machines (SVM)
A hyperplane that maximizes the distance between classes is used as the boundary for recognition.
```Python
classifier = EMGClassifier('SVM', data_set)
```

## Artificial Neural Networks (MLP)
A deep learning technique that uses human-like "neurons" to model data to help discriminate between inputs. Especially for this model, we **highly** recommend you create your own.
```Python
classifier = EMGClassifier('MLP', data_set)
```

## K-Nearest Neighbour (KNN)
Discriminates between inputs using the K closest samples in feature space. The implemented version in the toolkit defaults to k = 5.
```Python
classifier = EMGClassifier('KNN', data_set)
```

## Random Forest (RF)
Uses a combination of decision trees to discriminate between inputs.
```Python
classifier = EMGClassifier('RF', data_set)
```

## Quadratic Discriminant Analysis (QDA)
A statistical quadratic boundary is used to discriminate between inputs.
```Python
classifier = EMGClassifier('QDA', data_set)
```

## Naive Bayes (NB)
Similar to LDA, but has a more simplistic model of class covariances. 
```Python
classifier = EMGClassifier('NB', data_set)
```

## Gradient Boost (GB)
TODO: Evan
```Python
classifier = EMGClassifier('GB', data_set)
```