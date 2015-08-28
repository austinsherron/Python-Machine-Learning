################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np
from functools import reduce
from numpy import asmatrix as mat
from numpy import asarray as arr
from regress import Regress


################################################################################
################################################################################
################################################################################


################################################################################
## LINEARREGRESS ###############################################################
################################################################################


class LinearRegress(Regress):

	def __init__(self, X=None, Y=None, reg=0):
		"""
		Constructor for LinearRegressor (linear regression model).

		Args:
			X = N x M numpy array that contains N data points with M features
			Y = 1 x N numpy array that contains values the relate to the data
			  points in X
			reg = scalar (int or float) that is the L2 regularization penalty
			  ex: 1 / m ||y - w x'||^2 + reg * ||w||^2 
		"""
		self.theta = []

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(X, Y, reg)


	def __repr__(self):
		str_rep = 'LinearRegress, {} features\n{}'.format(
			len(self.theta), self.theta)
		return str_rep


	def __str__(self):
		str_rep = 'LinearRegress, {} features\n{}'.format(
			len(self.theta), self.theta)
		return str_rep


## CORE METHODS ################################################################
			

	def train(self, X, Y, reg=0):
		"""
		This method trains a linear regression predictor on the given data.
		Refer to the constructor doc string for argument descriptions.
		"""
		X_train = np.concatenate((np.ones((mat(X).shape[0],1)), X), axis=1)		# extend features by including a constant feature

		if reg == 0:
			if mat(X_train).shape[0] == mat(X_train).shape[1]:					# if number of data points == number of features...
				self.theta = np.linalg.solve(X_train, mat(Y).T)					# ...use 'solve' to solve the system
			else:
				self.theta = np.linalg.lstsq(X_train, mat(Y).T)					# ...else us 'lstsq' to find the least squares solution
			self.theta = mat(self.theta[0]).T									# make sure self.theta is a row, not a column
		else:
			m,n = mat(X_train).shape
			self.theta = mat(Y) * mat(X_train / m) * np.linalg.inv(mat(X_train).T * mat(X_train / m) + reg * np.eye(n))

		self.theta = arr(self.theta).ravel()									# make sure self.theta is flat


	def predict(self, X):
		"""
		This method makes a prediction on X using learned linear coefficients.

		Args:
			X = N x M numpy array that contains N data points with M features
		"""
		X_te = np.concatenate((np.ones((mat(X).shape[0],1)), X), axis=1)		# extend features by including a constant feature
		return mat(X_te) * mat(self.theta).T


## MUTATORS ####################################################################


	def set_weights(self, wts):
		"""
		This method sets the weights of linear regression model.

		Args:
			wts = list, numpy array/matrix that contains scalars (ints or floats)
			  that are the weights of the regressions model
		"""
		if (type(wts) not in [np.ndarray, np.matrix, list] or len(wts) == 0 or not 
			reduce(lambda x,y: x and y, map(lambda x: type(x) in [int, float], wts))):
			raise TypeError('LinearRegress.set_weights: wts must be an array/matrix/list that contains scalars (ints/floats)')
		self.theta = wts


## INSPECTORS ##################################################################


	def get_weights(self):
		return self.theta


## HELPERS #####################################################################


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	np.set_printoptions(linewidth=200)

	data = [[float(val) for val in row[:-1]] for row in csv.reader(open('../regressor-data.csv'))]
	trd = np.asarray(data[0:40] + data[50:90] + data[100:140])
	ted = np.asarray(data[40:50] + data[90:100] + data[140:150])
	trd2 = np.asarray(data[150:180] + data[200:230] + data[250:280])
	ted2 = np.asarray(data[180:200] + data[230:250] + data[280:300])
	trd3 = np.asarray(data[300:320] + data[350:370] + data[400:420])
	ted3 = np.asarray(data[320:350] + data[370:400] + data[420:450])
	predictions = [float(row[-1].lower()) for row in csv.reader(open('../regressor-data.csv'))]
	trp = np.asarray(predictions[0:40] + predictions[50:90] + predictions[100:140])
	tep = np.asarray(predictions[40:50] + predictions[90:100] + predictions[140:150])
	trp2 = np.asarray(predictions[150:180] + predictions[200:230] + predictions[250:280])
	tep2 = np.asarray(predictions[180:200] + predictions[230:250] + predictions[280:300])
	trp3 = np.asarray(predictions[300:320] + predictions[350:370] + predictions[400:420])
	tep3 = np.asarray(predictions[320:350] + predictions[370:400] + predictions[420:450])
	
	print('lr', '\n')
	lr = LinearRegress(trd, trp)
	print(lr, '\n')
	print(lr.predict(ted), '\n')
	print(lr.mae(ted, tep), '\n')
	print(lr.mse(ted, tep), '\n')
	print(lr.rmse(ted, tep), '\n')

	print()

	print('lr', '\n')
	lr = LinearRegress(trd, trp, reg=.4)
	print(lr, '\n')
	print(lr.predict(ted), '\n')
	print(lr.mae(ted, tep), '\n')
	print(lr.mse(ted, tep), '\n')
	print(lr.rmse(ted, tep), '\n')

	print('lr', '\n')
	lr = LinearRegress(trd, trp, reg=1.2)
	print(lr, '\n')
	print(lr.predict(ted), '\n')
	print(lr.mae(ted, tep), '\n')
	print(lr.mse(ted, tep), '\n')
	print(lr.rmse(ted, tep), '\n')

	print()

	print('lr', '\n')
	lr = LinearRegress(trd2, trp2, reg=1.6)
	print(lr, '\n')
	print(lr.predict(ted2), '\n')
	print(lr.mae(ted2, tep2), '\n')
	print(lr.mse(ted2, tep2), '\n')
	print(lr.rmse(ted2, tep2), '\n')

	print()

	print('lr', '\n')
	lr = LinearRegress(trd3, trp3, reg=0.6)
	print(lr, '\n')
	print(lr.predict(ted3), '\n')
	print(lr.mae(ted3, tep3), '\n')
	print(lr.mse(ted3, tep3), '\n')
	print(lr.rmse(ted3, tep3), '\n')

	print()

	print('lr', '\n')
	lr = LinearRegress(trd, trp, reg=3.2)
	print(lr, '\n')
	print(lr.predict(ted2), '\n')
	print(lr.mae(ted2, tep2), '\n')
	print(lr.mse(ted2, tep2), '\n')
	print(lr.rmse(ted2, tep2), '\n')

	print()

	print('lr', '\n')
	lr = LinearRegress(trd2, trp2, reg=2.6)
	print(lr, '\n')
	print(lr.predict(ted), '\n')
	print(lr.mae(ted, tep), '\n')
	print(lr.mse(ted, tep), '\n')
	print(lr.rmse(ted, tep), '\n')


################################################################################
################################################################################
################################################################################



