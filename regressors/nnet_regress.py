################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np
from numpy import asarray as arr
from numpy import asmatrix as mat
from regress import Regress


################################################################################
################################################################################
################################################################################


################################################################################
## NNETREGRESS #################################################################
################################################################################


class NNetRegress(Regress):

	def __init__(self, X=None, Y=None, sizes=[], init='zeros', stepsize=.01, tolerance=1e-4, max_steps=5000, activation='logistic'):
		"""
		Constructor for NNetRegressor (neural network classifier).
		Member weights are [W1 ... WL-1] where W1 is Nh1 x N1

		Args:
			X = N x M numpy array that contains N data points with M features
			Y = 1 x N numpy array that contains N values that relate to data points in X
			sizes = array of ints; [N1, Nh1 ... Nout] where Nout is # of outputs
			init = string; one of 'keep', 'zeros', or 'logistic; init method for weights'
			stepsize = scalar; step size for gradient descent (descreases as 1 / iter)
			tolerance = scalar; tolerance for stopping criterion
			max_steps = int; maximum number of steps before stopping
			activation = string; one of 'logistic', 'htangent', or 'custom'; init method for activation functions
		"""
		self.wts = arr([], dtype=object)
		self.activation = arr([])
		self.sig = arr([])
		self.d_sig = arr([])
		self.sig_0 = arr([])
		self.d_sig_0 = arr([])

		self.set_activation(activation)


	def __repr__(self):
		to_return = 'Multi-layer Perceptron (Neural Network) Regressor\n[{}]'.format(
			self.get_layers())
		return to_return


	def __str__(self):
		to_return = 'Multi-layer Perceptron (Neural Network) Regressor\n[{}]'.format(
			self.get_layers())
		return to_return


## CORE METHODS ################################################################


	def train(self, X, Y, init='zeros', stepsize=.01, tolerance=1e-4, max_steps=500):
		"""
		This method trains the neural network. Refer to constructor
		doc string for descriptions of arguments.
		"""
		pass


	def predict(self):
		pass


## MUTATORS ####################################################################


	def set_activation(self, method, sig=None, d_sig=None, sig_0=None, d_sig_0=None):
		pass


	def set_layers(self):
		pass


	def set_weights(self):
		pass


	def init_weights(self, sizes, init, X, Y):
		pass


## INSPECTORS ##################################################################


	def get_layers(self):
		pass


	def get_weights(self):
		pass


## HELPERS #####################################################################



################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	data = [[float(val) for val in row[:-1]] for row in csv.reader(open('../classifier-data.csv'))]
	trd = np.asarray(data[0:40] + data[50:90] + data[100:140])
	ted = np.asarray(data[40:50] + data[90:100] + data[140:150])
	classes = [float(row[-1].lower()) for row in csv.reader(open('../classifier-data.csv'))]
	trc = np.asarray(classes[0:40] + classes[50:90] + classes[100:140])
	tec = np.asarray(classes[40:50] + classes[90:100] + classes[140:150])

	btrd = trd[0:80,:]
	bted = ted[0:20,:]
	btrc = trc[0:80]
	btec = tec[0:20]

	btrd2 = trd[40:120,:]
	bted2 = ted[10:30,:]
	btrc2 = trc[40:120]
	btec2 = tec[10:30]

	print('nc')
	nc = NNetClassify(trd, trc, [2,5,3])
	print(nc)


################################################################################
################################################################################
################################################################################
