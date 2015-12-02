################################################################################
## IMPORTS #####################################################################
################################################################################


import matplotlib.pyplot as plt
import numpy as np
import random

from csv import reader
from data import bootstrap_data, load_data_from_csv, split_data
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod
from numpy import column_stack as cols
from classifiers.knn_classify import KNNClassify


################################################################################
################################################################################
################################################################################


################################################################################
##  PLOTTING FUNCTIONS #########################################################
################################################################################


def plot_classify_2D(learner, X, Y, pre=lambda x: x):
	"""
	Plot data and classifier outputs on two-dimensional data.
	This function plot data (X,Y) and learner.predict(X, Y) 
	together. The learner is is predicted on a dense grid
	covering data X, to show its decision boundary.

	Parameters
	----------
	learner : learner object
		A trained learner object that inherits from one of
		the 'Classify' or 'Regressor' base classes.
	X : numpy array
		N x M array of data; N = number of data, M = dimension
		(number of features) of data.
	Y : numpy array
		1 x N array containing labels corresponding to data points
		in X.
	pre : function object (optional)
		Function that is applied to X before prediction.
	"""
	if twod(X).shape[1] != 2:
		raise ValueError('plot_classify_2d: function can only be called using two-dimensional data (features)')

	plt.plot(X[:,0], X[:,1], 'k.')
	ax = plt.xlim() + plt.ylim()					# get current axis limits
	N = 256											# density of evaluation

	# evaluate each point of feature space and predict the class
	X1 = np.linspace(ax[0], ax[1], N)
	X1sp = twod(X1).T * np.ones(N)
	X2 = np.linspace(ax[2], ax[3], N)
	X2sp = np.ones((N,1)) * X2
	
	Xfeat = cols((twod(X1sp.flatten()).T, twod(X2sp.flatten()).T))

	# preprocess/create feature vector if necessary
	Xfeat = pre(Xfeat)

	# predict using learner
	pred = learner.predict(Xfeat)

	# plot decision values for space in 'faded' color
	clim = np.unique(Y)
	clim = [clim[0], clim[0] + 1] if len(clim) == 1 else list(clim)
	plt.imshow(np.reshape(pred, (N,N)).T, extent=[X1[0], X1[-1], X2[0], X2[-1]], cmap=plt.cm.Pastel2)
	plt.clim(*clim)

	plt.show()


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	X,Y = load_data_from_csv('../data/binary.csv', -1, float)
	X,Y = bootstrap_data(X, Y, 25)
	X = X[:,2:]
	Xtr,Xte,Ytr,Yte = split_data(X, Y, .8)
	knn = KNNClassify(Xtr, Ytr)

	print(cols((X,knn.predict(X))))
	
	plot_classify_2D(knn, X, Y)


################################################################################
################################################################################
################################################################################
