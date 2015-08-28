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
from sklearn.preprocessing import normalize


################################################################################
################################################################################
################################################################################


################################################################################
## LOGISTICREGRESS #############################################################
################################################################################


class LogisticRegress(Regress):

	def __init__(self, X=None, Y=None, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros'):
		"""
		Constructor for LogisticRegressor (logistic regression model).

		Args:
			X = N x M numpy array that contains N data points with M features
			Y = 1 x N numpy arra that contains values that relate to the data
			  points in X
			stepsize = scalar (int/float) that is the step size for gradient descent
			tolerance = scalar that is the tolerance for stopping criterion
			max_steps = int that is the maximum number of steps in gradient descent
			  before stopping
			init = 'zeros' or 'random'; 'zeros' initializes self.wts to vector of
			  0s, 'random' initializes self.wts to a vector of random numbers
		"""
		self.wts = []

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(X, Y, stepsize, tolerance, max_steps, init)


	def __repr__(self):
		str_rep = 'LogisticRegress, {} features\n{}'.format(
			len(self.wts), self.wts)
		return str_rep


	def __str__(self):
		str_rep = 'LogisticRegress, {} features\n{}'.format(
			len(self.wts), self.wts)
		return str_rep


## CORE METHODS ################################################################
			

	def train(self, X, Y, stepsize=.01, tolerance=1e-4, max_steps=5000, init='zeros'):
		"""
		This method trains the logistic regression model. See constructor
		doc string for argument descriptions.
		"""
		n,d = mat(X).shape
		X_train = np.concatenate((np.ones((n,1)), X), axis=1)

		self.wts = self.__init_wts(init, d)
		self.__gradient_descent(X, X_train, Y, n, max_steps, stepsize, tolerance)


	def predict(self, X):
		"""
		This method make a 'soft' prediction on X (predicts real 
		valued numbers). Refer to constructor for argument descriptions.
		"""
		return self.__logistic(X)


## MUTATORS ####################################################################


## INSPECTORS ##################################################################


## HELPERS #####################################################################


	def __gradient_descent(self, X, X_train, Y, n, max_steps, stepsize, tolerance):
		"""
		This is a helper method that implements stochastic gradient 
		descent. Used in:
			train
		"""
		iter, done = 1, 0
		mse, err = np.zeros((1,max_steps + 1)).ravel(), np.zeros((1,max_steps + 1)).ravel()

		while not done:
			Y_hat = self.__logistic(X)
			mse[iter] = np.mean(np.power(Y - Y_hat, 2))
			err[iter] = np.mean(Y != (Y_hat > .5))

			for i in range(n):
				Y_hat_i = self.__logistic(X[i,:])
				grad = (Y_hat_i - Y[i]) * Y_hat_i * (1 - Y_hat_i) * X_train[i,:]
				self.wts = self.wts - (stepsize / iter) * grad

			done = (iter > 1 and abs(mse[iter] - mse[iter - 1]) < tolerance) or iter > max_steps
			iter += 1

		self.wts = arr(self.wts).ravel()


	def __logistic(self, X):
		"""
		This is a helper method that evaluates the logistic function
		for weights self.wts (1 x d + 1) on data X (n x d). Used in:
			__gradient_descent
			predict
		"""
		n,d = mat(X).shape

		if n == 1:
			X = np.concatenate(([0], X), axis=1)
		else:
			X = np.concatenate((np.ones((n,1)), X), axis=1)

		f = mat(X) * mat(self.wts).T
		return 1 / (1 + np.exp(-f))


	def __init_wts(self, init, d):
		"""
		This is a helper method that initializes the wts of
		the regression model. Used in:
			train
		"""
		init = init.lower()

		if init == 'zeros':
			return np.zeros((1,d + 1)).ravel()
		elif init == 'random':
			return np.random.randn(1, d + 1).ravel()
		else:
			raise ValueError('LogisticRegress.__init_wts: ' + str(init) + ' is not a valid option for init')


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

	trdmu = np.mean(trd, axis=0)
	tedmu = np.mean(ted, axis=0)
	trdmax = np.max(trd, axis=0)
	tedmax = np.max(ted, axis=0)

	print(trdmu)
	print(tedmu)
	print(trdmax)
	print(tedmax)

	trd = trd - trdmu
	ted = ted - tedmu
	trd = trd / trdmax
	ted = ted / tedmax

	print(trd)
	print(ted)

	trd = normalize(trd, axis=1)
	ted = normalize(ted, axis=1)

	print(trd)
	print(ted)

	print('lr', '\n')
	lr = LogisticRegress(trd, trp)
	print(lr, '\n')
	print(lr.predict(ted), '\n')
	print(lr.mae(ted, tep), '\n')
	print(lr.mse(ted, tep), '\n')
	print(lr.rmse(ted, tep), '\n')

#	print()
#
#	print('lr', '\n')
#	lr = LogisticRegress(trd2, trp2)
#	print(lr, '\n')
#	print(lr.predict(ted2), '\n')
#	print(lr.mae(ted2, tep2), '\n')
#	print(lr.mse(ted2, tep2), '\n')
#	print(lr.rmse(ted2, tep2), '\n')



################################################################################
################################################################################
################################################################################



