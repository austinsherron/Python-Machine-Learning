################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np
from base_classify import BaseClassify
from classify import Classify


################################################################################
################################################################################
################################################################################


################################################################################
## NNETCLASSIFY ################################################################
################################################################################


class NNetClassify(Classify):

	def __init__(self, X, Y, sizes, init='zeros', stepsize=.01, tolerance=1e-4, max_steps=5000,
		activation='logistic'):
		"""
		Constructor for NNetClassifier (neural net classifier).

		Args:
			X is an N x M numpy array which contains N data points with M features.
			Y is a 1 x N numpy array which contains class labels that correspond
			  to the data points in X. 
			sizes = [Nin, Nh1, ... , Nout] where Nout is the number of outputs,
			  usually the number of classes. Member weights are {W1, ... , WL-1},
			  where W1 is Nh1 x Nin, etc.
			init is one of the following strings: none, zeros, or random which inits
			  the neural net weights.
			stepsize is the stepsize for gradient descent (decreases as 1 / iter).
			tolerance is the tolerance for stopping criterion.
			max_steps is the maximum number of steps before stopping
			activation is one of the following strings: logistic, htangent, or
			  custom, and it sets the activation functions.
		
		"""
		self.classes = []
		self.wts = np.asarray([], dtype=object)
		self.set_activation(activation.lower())
		self.init_weights(sizes, init.lower(), X, Y)

		if type(X) is np.ndarray and type(Y) is np.ndarray:
			self.train(X, Y, init, stepsize, tolerance, max_steps)


	def __repr__(self):
		to_return = 'Multi-layer Perceptron (Neural Network) Classifier\n[{}]'.format(
			self.get_layers())
		return to_return


	def __str__(self):
		to_return = 'Multi-layer Perceptron (Neural Network) Classifier\n[{}]'.format(
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


	def predict_soft(self):
		pass


	def err_k(self):
		pass


	def log_likelihood(self):
		pass


	def mse(self):
		pass


	def mse_k(self):
		pass


## MUTATORS ####################################################################


	def set_activation(self, method, sig=None, d_sig=None, sig_0=None, d_sig_0=None):
		"""
		This method sets the activation functions. 

		Args:
			method is one of the following string: logistic , htanged, or custom
			sig, d_sig, sig_0, and d_sig_0 are optional arguments intended for
			use with the 'custom' method option. They should be functions. sig_0
			and d_sig_0 are the output layer activation functions.
		"""
		method = method.lower()

		if method == 'logistic':
			self.sig = lambda z: 1 / (1 + np.exp(-z))
			self.d_sig = lambda z: sig(z) * (1 - sig(z))
			self.sig_0 = self.sig
			self.d_sig_0 = self.d_sig
		elif method == 'htangent':
			self.sig = lambda z: np.tanh(z)
			self.d_sig = lambda z: 1 - np.power(np.tanh(z), 2)
			self.sig_0 = self.sig
			self.d_sig_0 = self.d_sig
		elif method == 'custom':
			self.sig = sig
			self.d_sig = d_sig
			self.sig_0 = sig_0
			self.d_sig_0 = d_sig_0
		else:
			raise ValueError('NNetClassify.set_activation: ' + str(method) + ' is not a valid option for method')

		self.activation = method


	def set_classes(self):
		pass


	def set_layers(self):
		pass


	def set_weights(self):
		pass


	def init_weights(self, sizes, init, X, Y):
		"""
		This method initializes the weights of the neural network and
		sets layer sizes to S=[Ninput, N1, N2, ... , Noutput]

		Args:
			Refer to constructor doc string for description of sizes.
			init is one of the following strings: none, zeros, or random.
			Refer to constructor doc string for descriptions of X and Y.
		"""
		init = init.lower()

		if init == 'none':
			pass
		elif init == 'zeros':
			self.wts = np.asarray([np.zeros((sizes[i + 1],sizes[i] + 1)) for i in range(len(sizes) - 1)], dtype=object)
		elif init == 'random':
			self.wts = np.asarray([.25 * np.random.randn(sizes[i+1],sizes[i]+1) for i in range(len(sizes) - 1)], dtype=object)
		else:
			raise ValueError('NNetClassify.init_weights: ' + str(init) + ' is not a valid option for init')


## INSPECTORS ##################################################################


	def get_classes(self):
		pass


	def get_layers(self):
		S = np.asarray([np.asmatrix(self.wts[i]).shape[1] - 1 for i in range(len(self.wts))])
		S = np.concatenate((S, [np.asmatrix(self.wts[-1]).shape[0]]))
		return S


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
