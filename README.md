# Python Machine Learning Library

These machine learning algorithms have been adapted from
algorithms written in Matlab by Professor [Alex Ihler](http://www.ics.uci.edu/~ihler/). 

## Documentation

### Regressors

#### Regress

```Regress``` is the abstract base class for all regressors, 
and implements methods that generalize to all other regressors. 
```Regress``` also implements a ```__call__``` method that provides
syntatic sugar for ```train``` and ```predict``` methods.

##### __call__

```python
def __call__(self, *args, **kwargs)
```
To predict: ```regressor(X)``` == ```regressor.predict(X)```. The first arg 
should be a numpy array, other arguments can be keyword args as necessary.
To train: ```regressor(X, Y, **kwargs)``` == ```regressor.train(X, Y, **kwargs)```.
The first and second args should be numpy arrays, other arguments can be keyword args 
as necessary.

###### Args 

This method takes any number of args or keyword args, the first two being numpy arrays.

##### mae

```python
def mae(self, X, Y)
```
This method computes the mean absolute error of predictor object on test data X and Y.

###### Args

X = N x M numpy array that contains N data points with M features
<br>
Y = 1 x N numpy array that contains values that correspond to the data points in X

##### mse

```python
def mse(self, X, Y)
```
This method computes the mean squared error of predictor object on test data X and Y. 

###### Args

X = N x M numpy array that contains N data points with M features
<br>
Y = 1 x N numpy array that contains values that correspond to the data points in X

##### rmse

```python
def rmse(self, X, Y)
```
This method computes the root mean squared error of predictor object on test data X and Y. 

###### Args

X = N x M numpy array that contains N data points with M features
<br>
Y = 1 x N numpy array that contains values that correspond to the data points in X

####


## Todos

#### General


* fix ```auc/roc``` methods, retest
* ~~fix ```linreg``` option in ```LinearClassify``` and test~~
* test ```train_soft``` in ```LinearClassify```/```LogisticClassify```
* make sure all methods implemented/retest
* implement ```nnetClassify``` 
* test ```TreeClassify```training options 
* ~~change ```range``` to ```permutation``` (```grep 'np.random.permutation*'```) and retest~~
* add plotting 
* fix ```LogisticRegress```
* fix ```BaggedClassify``` train method
* ~~fix ```BaggedClassify.__setitem__```~~	

#### Next Steps

* ~~finish and test ```baggedClassify```~~
* implement and test ```logisticMseClassify```
* implement and test ```nnetClassify```
* ~~implement and test ```knnRegress```~~
* ~~implement and test ```linearRegress```~~
* ~~implement and test ```treeRegress```~~
* ~~implement and test ```baggedRegress```~~
* implement and test ```nnetRegress```
* implement and test ```logisticRegress```

#### Low Priority

* add Ihler's comments 
* execute comprehensive tests
* ensure consistency of doc strings
* ~~fix indentation in regressors~~
* arg error checking
* modularize ```__dectree_train``` in ```TreeClassifer```
* ~~make sure inheritance is optimally utilized while maintaing clarity (added ```to_1_of_K``` to ```Classify```)~~


## Potential Bugs

* ```Y``` (class labels/data values) is flat 1 x N array
* generally use flat array instead of one row arrays ([0] vs [[0]]) or column vectors
* predictions are returned as columns
* python indices start at 0, matlab indices start at 1; inconsistent use of both in these tools 
* ~~some classifiers can't be retrained, must instantiate new object~~
* ~~train methods don't have default args~~ 
* ```.T``` notation doesn't work on flat arrays



