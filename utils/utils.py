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
## UTILITY FUNCTIONS ###########################################################
################################################################################


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
	Function that converts Y from 1-of-k rep back to single col/row form.

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


def to_index(Y, values=None):
	"""
	Function that converts discrete value Y into [0 .. K - 1] (index) 
	representation; i.e.: to_index([4 4 1 1 2 2], [1 2 4]) = [2 2 0 0 1 1].

	Parameters
	----------
	Y : array
		1 x N (N x 1) array of values to be converted.
	values : list (optional)
		List that specifices the value/index mapping to use for conversion.

	Returns
	-------
	idx : array
		Array that contains indexes instead of values.
	"""
	n,d = np.matrix(Y).shape

	assert min(n,d) == 1
	values = values if values else list(np.unique(Y))
	C = len(values)
	flat_Y = Y.flatten()

	idx = []
	for v in Y:
		idx.append(values.index(v))
	return np.array(idx)


def from_index(Y, values):
	"""
	Convert index-valued Y into discrete representation specified by values
	in values.

	Parameters
	----------
	Y : numpy array
		1 x N (or N x 1) numpy array of indices.
	values : numpy array
		1 x max(Y) array of values for conversion.

	Returns
	-------
	discrete_Y : numpy array
		1 x N (or N x 1) numpy array of discrete values.
	"""
	discrete_Y = values[Y]
	return discrete_Y


def optional_return(to_return, *args):
	"""
	Helper that allows optional return of arguments.

	Parameters
	----------
	to_return : [bool]
		The bool at each index indicates whether the value at that index in 
		args should be returned; e.g.: if to_return[0] == True return args[0].
	args : mixed
		All possible return values.

	Returns
	-------
	d : tuple (or single value)
		Tuple of returned arguments or single return value if there is only one
		return value.
	"""
	d = zip(to_return, args)
	d = tuple((map(lambda x: x[1], filter(lambda x: x[0], d))))

	if len(d) == 1:
		return d[0]
	else:
		return d


################################################################################
################################################################################
################################################################################
