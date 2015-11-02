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
## DATA MANIPULATION/CREATION  FUNCTIONS #######################################
################################################################################


def load_data_from_csv(csv_path, label_index, trans_func=lambda x: x):
	"""
	Function that loads from a CSV into main memory.

	Parameters
	----------
	csv_path : str
		Path to CSV file that contains data.
	label_indes : int
		The index in the CSV rows that contains the label
		for each data point.
	trans_func : function object
		Function that transform values in CSV, i.e.: str -> int.

	Returns
	-------
	data,labels : (list)
		Tuple that contains a list of data points (index 0) and
		a list of labels corresponding to thos data points (index 1).
	"""
	data = []
	labels = []

	with open(csv_path) as f:
		csv_data = reader(f)
	
		for row in csv_data:
			row = list(map(trans_func, row))

			labels.append(row.pop(label_index))
			data.append(row)

	return data,labels


def filter_data(data, labels, filter_func):
	"""
	Function that filters data based on filter_func. Function
	iterates through data and labels and passes the values
	produced by the iterables to filter_func. If filter_func
	returns True, the values aren't included in the return
	arrays.

	Parameters
	----------
	data : array-like
		Array that contains data points.
	labels : array-like
		Array that contains labels.
	filter_func : function object
		Function that filters data/labels.

	Returns
	-------
	filtered_data,filtered_labels : (list)
		Filtered arrays.
	"""
	filtered_data,filtered_labels = [], []
	for point,label in zip(data,labels):
		if not filter_func(point,label):
			filtered_data.append(point)
			filtered_labels.append(label)

	return filtered_data,filtered_labels


def bootstrap_data(X, Y, n_boot):
	"""
	Function that resamples (bootstrap) data set: it resamples 
	data points (x_i,y_i) with replacement n_boot times.

	Parameters
	----------
	X : numpy array
		N x M numpy array that contains data points to be sampled.
	Y : numpy array
		1 x N numpy arra that contains labels that map to data 
		points in X.
	n_boot : int
		The number of samples to take.

	Returns
	-------
	(array,array)
		Tuple containing samples from X and Y.
	"""
	nx,dx = twod(X).shape
	idx = np.floor(np.random.rand(n_boot) * nx).astype(int)
	X = X[idx,:]

	ny = len(Y)
	assert ny > 0, 'bootstrap_data: Y must contain data'
	assert nx == ny, 'bootstrap_data: X and Y should have the same length'
	Y = Y[idx]

	return (X,Y)


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':


	print('testing bootstrap_data')
	print()

	n,d = 100, 5
	n_boot = 30

	X = arr([np.random.rand(d) * 25 for i in range(n)])
	Y = np.floor(np.random.rand(n) * 3)
	data,classes = bootstrap_data(X, Y, n_boot)

	assert len(data) == len(classes) == n_boot
	assert d == twod(X).shape[1]

	print('data')
	print(data)
	print('classes')
	print(classes)

	print()
	print()


################################################################################
################################################################################
################################################################################
