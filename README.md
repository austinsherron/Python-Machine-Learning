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

```python
def __call__(self, *args, **kwargs)
```

To predict: ```regressor(X)``` == ```regressor.predict(X)```. The first arg 
should be a numpy array, other arguments can be keyword args as necessary.
To train: ```regressor(X, Y, **kwargs)``` == ```regressor.train(X, Y, **kwargs)```.
The first and second args should be numpy arrays, other arguments can be keyword args 
as necessary.

Args: this method takes any number of args or keyword args, the first two being numpy arrays.


## Todos

#### General


* fix ```auc/roc``` methods, retest
* fix ```linreg``` option in ```LinearClassify``` and test
* test ```train_soft``` in ```LinearClassify```/```LogisticClassify```
* make sure all methods implemented/retest
* implement ```nnetClassify``` 
* test ```TreeClassify```training options 
* ~~DONE: change ```range``` to ```permutation``` (```grep 'np.random.permutation*'```) and retest~~
* add plotting 
* fix ```LogisticRegress```
* fix ```BaggedClassify``` train method
* ~~DONE: fix ```BaggedClassify.__setitem__```~~	

#### Next Steps

* ~~DONE: finish and test ```baggedClassify```~~
* implement and test ```logisticMseClassify```
* implement and test ```nnetClassify```
* ~~DONE: implement and test ```knnRegress```~~
* ~~DONE: implement and test ```linearRegress```~~
* ~~DONE: implement and test ```treeRegress```~~
* ~~DONE: implement and test ```baggedRegress```~~
* implement and test ```nnetRegress```

#### Low Priority

* add Ihler's comments 
* execute comprehensive tests
* ensure consistency of doc strings
* ~~DONE: fix indentation in regressors~~
* arg error checking
* modularize ```__dectree_train``` in ```TreeClassifer```
* ~~DONE(?): make sure inheritance is optimally utilized while maintaing clarity (added ```to_1_of_K``` to ```Classify```)~~


## Potential Bugs

* ```Y``` (class labels/data values) is flat 1 x N array
* generally use flat array instead of one row arrays ([0] vs [[0]]) or column vectors
* predictions are returned as columns
* python indices start at 0, matlab indices start at 1; inconsistent use of both in these tools 
* ~~FIXED(?): some classifiers can't be retrained, must instantiate new object~~
* ~~FIXED(?): train methods don't have default args~~ 
* ```.T``` notation doesn't work on flat arrays



