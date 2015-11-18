################################################################################
## IMPORTS #####################################################################
################################################################################


import data
import matplotlib.pyplot as plt
import numpy as np
import random

from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod


################################################################################
################################################################################
################################################################################


################################################################################
## EXPECTATION-MAXIMIZATION ####################################################
################################################################################


def em_cluster(X, K, init='random', max_iter=100, tol=1e-6, do_plot=False):
	"""
	Perform Gaussian mixture EM (expectation-maximization) clustering on data X.

	Parameters
	----------
	X : numpy array
		N x M array containing data to be clustered.
	K : int
		Number of clusters.
	init : str or array (optional)
		Either a K x N numpy array containing initial clusters, or
		one of the following strings that specifies a cluster init
		method: 'random' (K random data points (uniformly) as clusters),
		'farthest' (choose cluster 1 uniformly, then the point farthest
		from all cluster so far, etc.), or 'k++' (choose cluster 1 
		uniformly, then points randomly proportional to distance from
		current clusters).
	max_iter : int
	tol : scalar

	Returns
	-------

	"""
	N,D = twod(X).shape					# get data size


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	X,Y = data.load_data_from_csv('../classifier-data.csv', 4, float)
	X,Y = arr(X), arr(Y)


################################################################################
################################################################################
################################################################################
