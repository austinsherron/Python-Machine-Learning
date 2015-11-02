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
## TESTING FUNCTIONS ###########################################################
################################################################################


def test_randomly(data, labels, mix=0.8, test=lambda x: 1.0, *args):
	"""
	Function that performs random tests using data/labels.

	Parameters
	----------
	data : numpy array
		N x M array of data points used for training/testing learner.
		N = number of data; M = number of features.
	labels : numpy array
		1 x N array of class/regression labels used for training/testing learner.
	mix : float
		The percentage of data to use for training (1 - mix = percentage of data
		used for testing).
	test : function object
		A function that takes at least four arguments (arrays containing data/labels
		for testing/training) and performs tests. This function should return an
		error value for one experiment.
	args : mixed
		Any additional arguments needed for testing.

	Returns
	-------
	float
		Average error value of all tests performed.
	"""
	start = 0
	end = len(data)

	avg_err = 0

	for i in range(start, end):
		indexes = range(end)
		train_indexes = random.sample(indexes, int(mix * end))
		test_indexes = list(set(indexes) - set(train_indexes))

		trd,trc = data[train_indexes], labels[train_indexes]
		ted,tec = data[test_indexes], labels[test_indexes]
		avg_err += test(trd, trc, ted, tec, *args)

	return avg_err / end


################################################################################
################################################################################
################################################################################
