################################################################################
## IMPORTS #####################################################################
################################################################################


import numpy as np

from csv import reader


################################################################################
################################################################################
################################################################################


################################################################################
## UTILITY FUNCTIONS ###########################################################
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


def to_1_of_k(Y, values=None):
	"""
	Function that converts Y into discrete valued matrix;
	i.e.: to_1_of_k([3,3,2,2,4,4]) = [[ 1 0 0 ]
									  [ 1 0 0 ]
									  [ 0 1 0 ]
									  [ 0 1 0 ]
									  [ 0 0 1 ]
									  [ 0 0 1 ]]

	Parameters
	----------
	Y : array like
		1 x N (or N x 1) array of values (ints) to be converted.
	values : list (optional)
		List that specifices indices of of Y values in return matrix.

	Returns
	-------
	array
		Discrete valued 2d representation of Y.
	"""
	n,d = np.matrix(Y).shape

	assert min(n,d) == 1
	values = values if values else list(np.unique(Y))
	C = len(values)
	flat_Y = Y.flatten()
	
	index = []
	for l in flat_Y:
		index.append(values.index(l))

	return np.array([[0 if r != i else 1 for i in range(C)] for r in index])


def from_1_of_k(Y, values=None):
	"""
	Funciton that converts Y from 1-of-k rep back to single col/row form.

	Parameters
	----------
	Y : arraylike
		Matrix to convert from 1-of-k rep.
	values : list (optional)
		List that specifies which values to use for which index.

	Returns
	-------
	array
		Y in single row/col form.
	"""
	return Y.argmax(1) if not values else np.atleast_2d([values[i] for i in Y.argmax(1)]).T


################################################################################
################################################################################
################################################################################
