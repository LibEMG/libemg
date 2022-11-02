# Selection Strategies
For a robust summary of the following feature selection techniques, please see *A Multivariate Approach for Predicting Myoelectric Control Usability* <sup>[1]</sup>.

## **Classification Accuracy (CA)**
Emphasizes classification accuracy of a classifier using leave-one-trial-out cross validation.

$
\text{CA} = \frac{1}{T}\sum_{t=1}^{n}(\frac{1}{n}\sum_{i=1}^{n}\hat{y}_{i,t} == {y}_{i,t})
$

where $T$ is the total number of trials, $t$ is an individual trial, $n$ is the total number of data frames, $\hat{y}_{i,t}$ is the predicted class label for frame $i$, and $y_{i,t}$ is the true class label for frame i.

## **Active Error (AE)**
Similar to CA but focuses on active error (i.e., the error of the classifier without considering incorrect No Movmenet predictions). Also known as (1 - active classification accuracy).

$
\text{AE} = \frac{1}{T}\sum_{t=1}^{n}(1-(\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i,t} == {y}_{i,t}) \text{ or } (\hat{y}_{i,t} == {y}_{NM})))
$

where $T$ is the total number of trials, $t$ is an individual trial, $n$ is the total number of data frames, $\hat{y}_{i,t}$ is the predicted class label for frame $i$, $y_{i,t}$ is the true class label for frame $i$ and $y_{NM}$ is the no movement class.

## **Mean Semi Principle Axis (MSA)**
Quantifies the size of a training elipsoid.

$
\text{MSA} = \frac{1}{N}\sum_{j=1}^{N}((\prod_{k=1}^{D}a_{j,k})^{\frac{1}{D}})
$

where $N$ is the number of classes, $j$ represents a specific class, $D$ is the total dimensionality of the space, and $a_{k}$ is the geometric mean of each semi-principle axis.

## **Feature Efficiency (FE)**
A measure of the fraction of samples seperable by a particular feature.

$
\text{FE} = \frac{1}{N}\sum_{j=1}^{N}\max\limits_{i=1,...,j-1,j+1,...,N} \\ \times (\max\limits_{k=1,...,D} \frac{n(C_{i}) + n(C_{j}) - n(S_{k})}{n(C_{i}) + n(C_{j})})
$

$
S_{k} = p | p \space \epsilon \space C_{i} \space \cup \space C_{j}, \space \min(\max(f_{k}|c_{j}), \max(f_{k}|c_{i})) \ge p \ge \max(\min(f_{k}|c_{j}), \min(f_{k}|c_{i}))
$

where $N$ is the number of active classes, $j$ and $i$ are particular class labels, $D$ is the dimensionality, $n(C_{i})$ is the cardinality of the set of data points in class $i$ and $n(C_{j})$ in class $j$, $S_{k}$ is the set of points not seperable along a feature dimension $k$, $n(S_{k})$ is the cardinality of the overlap set, $f_{k}|c_{i}$ is the value of feature $f$ in dimension $k$ for class label $i$, and $p$ is the $D$ dimensional data point in class $i$ or $j$.

# References
<a id="1">[1]</a> 
J. L. Nawfel, K. B. Englehart and E. J. Scheme, "A Multi-Variate Approach to Predicting Myoelectric Control Usability," in IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 29, pp. 1312-1327, 2021, doi: 10.1109/TNSRE.2021.3094324.