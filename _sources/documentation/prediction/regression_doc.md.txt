# Regressors

Below is a list of the regressors that can be instatiated by passing in a string to the `EMGRegressor`. For other regressors, pass in a custom model that has the `fit` and `predict` methods.

## Linear Regression (LR)

A linear regressor that aims to minimize the residual sum of squares between the predicted values and the true targets.

```Python
regressor = EMGRegressor('LR')
regressor.fit(data_set)
```

Check out the LR docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

## Support Vector Machines (SVM)

A regressor that uses a kernel trick to find the hyperplane that best fits the data.

```Python
regressor = EMGRegressor('SVM')
regressor.fit(data_set)
```

Check out the SVM docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

## Artificial Neural Networks (MLP)

A deep learning technique that uses human-like "neurons" to model data to help discriminate between inputs. Especially for this model, we **highly** recommend you create your own.

```Python
regressor = EMGRegressor('MLP')
regressor.fit(data_set)
```

Check out the MLP docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)

## Random Forest (RF)

Uses a combination of decision trees to discriminate between inputs.

```Python
regressor = EMGRegressor('RF')
regressor.fit(data_set)
```

Check out the RF docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

## Gradient Boosting (GB)

Additive model that fits a regression tree on the negative gradient of the loss function.

```Python
regressor = EMGRegressor('GB')
regressor.fit(data_set)
```

Check out the GB docs [here.](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
