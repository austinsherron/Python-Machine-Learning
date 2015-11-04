################################################################################
## IMPORTS #####################################################################
################################################################################


import numpy as np
import random

from csv import reader
from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod


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
		1 x N arra containing labels corresponding to data points
		in X.
	pre : function object (optional)
		Function that is applied to X before prediction.
	"""
	if twod(X).shape[1] != 2:
		raise ValueError('plot_classify_2d: function can only be called using two-dimensional data (features)')


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	pass


################################################################################
################################################################################
################################################################################
