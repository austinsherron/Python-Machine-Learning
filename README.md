# Python Machine Learning Library

These machine learning algorithms have been adapted from
algorithms written in Matlab by Professor [Alex Ihler](http://www.ics.uci.edu/~ihler/). 
All functions/objects include docstrings that specify parameters and return values.

## Contents

### Classifiers

- Bagged Classifier (`BaggedClassify`)
- Gauss-Bayes Classifier (`GaussBayesClassify`)
- K-Nearest Neighbor Classifier (`KNNClassify`)
- Linear Classifier (`LinearClassify`)
- Logistic Classifier (`LogisticClassify`)
- Logistic Mean Squared Error Classifier (`LogisticMSEClassify`)
- Neural Net Classifier (`NNetClassify`)
- Decision Tree Classifier (`TreeClassify`)

### Regressors

- Bagged Regressor (`BaggedRegress`)
- K-Nearest Neighbor Regressor (`KNNRegress`)
- Linear Regressor (`LinearRegress`)
- Logistic Regressor (`LogisticRegress`)
- Neural Net Regressor (`NNetRegress`)
- Decision Tree Regressor (`TreeRegress`)

### Unsupervised Learners (Clustering)

- Hierarchical Agglomerative Clustering (`agglom_cluster`)
- Expectation-Maximization Clustering (`em_cluster`)
- K-Means Clustering (`kmeans`)

### Boosting 

- Gradient Boosting (`GradBoost`)
- Adaptive Boosting (`AdaBoost`)

### Utilities

#### Data Generation/Manipulation 

- `load_data_from_csv`
- `filter_data`
- `bootstrap_data`
- `data_GMM`
- `gamrand`
- `data_gauss`
- `whiten`
- `split_data`
- `shuffle_data`
- `rescale`

#### Feature Extraction

- `fhash`
- `fkitchensink`
- `flda`
- `fpoly`
- `fpoly_mono`
- `fpoly_pair`
- `fproject`
- `fsubset`
- `fsvd`

#### Testing/Learner Validation

- `cross_validate`
- `test_randomly`

#### Misc. Utilities

- `to_1_of_k`
- `from_1_of_k`
- `to_index`
- `from_index`
- `optional_return`

## Todos

### General

- ~~fix `auc/roc` methods~~, retest
- ~~remove dependency on BaseClassify~~
- ~~fix `linreg` option in `LinearClassify` and test~~
- test `train_soft` in `LinearClassify`/`LogisticClassify`
- test `TreeClassify`training options 
- ~~change `range` to `permutation` (`grep 'np.random.permutation\*'`) and retest~~
- add plotting 
- fix `LogisticRegress`
- ~~fix `BaggedClassify` train method~~
- ~~fix `BaggedClassify.__setitem__`~~	
- test `logisticMseClassify`

### Next Steps

- ~~finish and test `baggedClassify`~~
- ~~implement `logisticMseClassify`~~
- implement and test `nnetClassify`
- ~~implement and test `knnRegress`~~
- ~~implement and test `linearRegress`~~
- ~~implement and test `treeRegress`~~
- ~~implement and test `baggedRegress`~~
- implement and test `nnetRegress`
- ~~implement and test `logisticRegress`~~
- add plotting
- finish helpers
- add Ihler's comments 

### Low Priority

- ensure consistency of doc strings
- ~~fix indentation in regressors~~
- arg error checking
- modularize `__dectree_train` in ```TreeClassifer```
- ~~make sure inheritance is optimally utilized while maintaing clarity (added `to_1_of_K` to `Classify`)~~
- classify.to_1_of_k is wrong, but tree/logisitc classify depend on it; make those
  classifiers work with the correct version in utils

## Potential Bugs

- `Y` (class labels/data values) is flat 1 x N array
- generally use flat array instead of one row arrays ([0] vs [[0]]) or column vectors
- predictions are returned as columns
- python indices start at 0, matlab indices start at 1; inconsistent use of both in these tools 
- ~~some classifiers can't be retrained, must instantiate new object~~
- ~~train methods don't have default args~~ 
- `.T` notation doesn't work on flat arrays

