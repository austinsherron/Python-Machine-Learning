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
## GAUSSBAYESCLASSIFY ##########################################################
################################################################################


class GaussBayesClassify(BaseClassify):

	def __init__(self, X=None, Y=None, equal=0, diagonal=0, wts=None, reg=0):
		"""
		Constructor for GaussBayesClassifier. X and Y must be numpy arrays.
		X is an N x M array that contains N data points with M features.
		Y is a 1 x N array that contains the classes that correspond to the
		data points in X. equal is a bool (or equivalent) that forces all 
		classes to share a single covariance model. diagonal is a bool 
		(or equivalent) that forces all classes to use a diagonal covariance
		model. wts is a 1 x M vector of positive weights (ints or floats).
		reg is an int or float that regularizes the covariance model.
		"""
		self.means = []
		self.covars = []
		self.probs = []
		self.classes = []

		if type(X) == np.ndarray and type(Y) == np.ndarray:
			self.train(X, Y, equal, diagonal, wts, reg)


	def __repr__(self):
		to_print = 'Gaussian classifier, {} classes:\n{}\nMeans:\n{}\nCovariances:\n{}\n'.format(
			len(self.classes), self.classes, 
			str([str(np.asmatrix(m).shape[0]) + ' x ' + str(np.asmatrix(m).shape[1]) for m in self.means]), 
			str([str(np.asmatrix(c).shape[0]) + ' x ' + str(np.asmatrix(c).shape[1]) for c in self.covars])) 
		return to_print

	
	def __str__(self):
		to_print = 'Gaussian classifier, {} classes:\n{}\nMeans:\n{}\nCovariances:\n{}\n'.format(
			len(self.classes), self.classes, 
			str([str(np.asmatrix(m).shape[0]) + ' x ' + str(np.asmatrix(m).shape[1]) for m in self.means]), 
			str([str(np.asmatrix(c).shape[0]) + ' x ' + str(np.asmatrix(c).shape[1]) for c in self.covars])) 
		return to_print


## CORE METHODS ################################################################


	def train(self, X, Y, equal=0, diagonal=0, wts=None, reg=0):
		"""
		This method trains a Bayes classifier with class models. Refer to 
		the constructor doc string for descriptions of X and Y.
		"""
		wts = wts if type(wts) == np.ndarray else [1 for i in range(len(Y))]
		wts = np.divide(wts, np.sum(wts))

		self.classes = list(np.unique(Y)) if type(Y) == np.ndarray else []

		for i in range(len(self.classes)):
			indexes = np.where(Y == self.classes[i])[0]	
			self.probs.insert(i, np.sum(wts[indexes]))
			wtsi = np.divide(wts[indexes], self.probs[i])
			self.means.insert(i, np.dot(wtsi, X[indexes,:]))
			tmp = X[indexes,:] - self.means[i]
			wtmp = np.transpose(wtsi * X[indexes,:].T)

			self.__set_covars(tmp, wtmp, i, diagonal, reg)

		if equal:
			self.__handle_equal_covar()


	def predict_soft(self, X):
		"""
		This method makes "soft" predictions on test data using the trained
		model. X is an N x M matrix of N data points with M features. N
		doesn't necessarily have to be the same value as N in the training
		method.
		"""
		m = np.shape(np.asmatrix(X))[0]
		C = len(self.classes)
		p = np.zeros((m, C))
		for c in range(C):
			p[:,c] = self.probs[c] * self.__eval_gaussian(X, self.means[c], self.covars[c])
		p = p / np.tile(np.transpose(np.asmatrix(np.sum(p, axis=1))), (1,C))
		return np.asarray(p)


	def predict(self, X):
		"""
		This method makes predictions on test data using the trained
		model. Refer to the predict_soft doc string for a description of X.
		"""
		p = np.asmatrix(self.predict_soft(X))
		max_i = np.argmax(p, axis=1)
		return np.asarray([[self.classes[r[0]]] for r in max_i])


	def mae(self, X, Y):
		"""
		This method calculates the absolute mean error for a given validation
		data set. Refer to constructor doc string for descriptions of X and Y.
		"""
		n,m = np.asmatrix(Y).shape
		Y_hat = self.predict(X).ravel() if n < m else self.predict(X)
		err = np.mean(np.abs(Y_hat - Y))
		return err


	def mse(self, X, Y):
		"""
		This method calculates the mean squared error for a given validation
		data set. Refer to constructor doc string for descriptions of X and Y.
		"""
		n,m = np.asmatrix(Y).shape
		Y_hat = self.predict(X).ravel() if n < m else self.predict(X)
		err = np.mean(np.power(Y_hat - Y, np.asarray([2 for i in range(len(Y))])))
		return err
		

## MUTATORS ####################################################################


	def set_classes(self, classes):
		"""
		Set classes of the classifier. classes should
		be a list. 
		"""
		if type(classes) is not list or len(classes) == 0:
			raise TypeError('classes must be a list with a length of at least 1')
		self.classes = classes


	def set_covars(self, covars):
		"""
		Set covariances of the classifier. covars should be a
		list of numpy arrays or matrices.
		"""
		if type(covars) is not list or len(covars) == 0:
			raise TypeError('covars must be a list with a length of at least 1 that contains numpy arrays or matrices')
		self.covars = covars


	def set_means(self, means):
		"""
		Set means of the classifier. means should be a list
		of numpy arrays or matrices.
		"""
		if type(means) is not list or len(means) == 0:
			raise TypeError('means must be a list with a length of at least 1 that contains numpy arrays or matrices')
		self.means = means


	def set_probs(self, probs):
		"""
		Set probs of the classifier. probs should be a list.
		"""
		if type(probs) is not list or len(probs) == 0:
			raise TypeError('probs must be a list with a length of at least 1')
		self.probs = probs



## INSPECTORS ##################################################################


	def get_classes(self):
		return self.classes


	def get_covars(self):
		return self.covars


	def get_means(self):
		return self.means


	def get_probs(self):
		return self.probs


## HELPERS #####################################################################


	def __set_covars(self, tmp, wtmp, i, diagonal, reg):
		"""
		This is a helper method that calculates covariances.  Used in:
			train
		"""
		if diagonal:
			self.covars.insert(i, np.diag(sum(tmp * wtmp) + reg))
		else:
			self.covars.insert(i, np.dot(tmp.T, wtmp) + np.diag(reg + 0 * self.means[i]))	


	def __handle_equal_covar(self):
		"""
		This is a helper method that handles the equal covariance option.  Used in:
			train
		"""
		call = 0
		for i in range(len(self.classes)):
			call += self.probs[i] * self.covars[i]
		for i in range(len(self.classes)):
			self.covars[i] = call


	def __eval_gaussian(self, X, means, covars):
		"""
		This is a helper method that helps calculate probabilities according to 
		Bayes Rules.  Used in:
			predict_soft
		"""
		n = np.shape(np.asmatrix(X))[0]
		d = np.shape(np.asmatrix(X))[1]
		p = np.zeros((n, 1))
		constant = 1 / (2 * math.pi)**(d / 2) / np.linalg.det(covars)**(0.5)
		inverse = np.linalg.inv(covars)
		R = X - np.tile(means, (n, 1))
		p = np.exp(-0.5 * np.sum(np.dot(R, inverse) * R, axis=1)) * constant
		return p
			
		
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

	print('bgbc', '\n')
	bgbc = GaussBayesClassify(btrd, btrc, equal=0, diagonal=0, wts=None, reg=0)
	print(bgbc, '\n')
	print(bgbc.predict(bted), '\n')
	print(bgbc.predict_soft(bted), '\n')
	print(bgbc.auc(bted, btec), '\n')
	print(bgbc.err(bted, btec), '\n')
	print(bgbc.roc(bted, btec), '\n')

	print()

	print('bgbc2', '\n')
	bgbc2 = GaussBayesClassify(btrd, btrc, equal=1, diagonal=1, wts=None, reg=np.asarray([1,3,4,6]))
	print(bgbc2, '\n')
	print(bgbc2.predict(bted), '\n')
	print(bgbc2.predict_soft(bted), '\n')
	print(bgbc2.auc(bted, btec), '\n')
	print(bgbc2.err(bted, btec), '\n')
	print(bgbc2.roc(bted, btec), '\n')

	print()

	print('ibgbc', '\n')
	ibgbc = GaussBayesClassify(bted, btec, equal=0, diagonal=0, wts=None, reg=0)
	print(ibgbc, '\n')
	print(ibgbc.predict(btrd), '\n')
	print(ibgbc.predict_soft(btrd), '\n')
	print(ibgbc.auc(btrd, btrc), '\n')
	print(ibgbc.err(btrd, btrc), '\n')
	print(ibgbc.roc(btrd, btrc), '\n')

	print()

	print('bgbc3', '\n')
	bgbc3 = GaussBayesClassify(btrd2, btrc2, equal=0, diagonal=0, wts=None, reg=0)
	print(bgbc3, '\n')
	print(bgbc3.predict(bted2), '\n')
	print(bgbc3.predict_soft(bted2), '\n')
	print(bgbc3.auc(bted2, btec2), '\n')
	print(bgbc3.err(bted2, btec2), '\n')
	print(bgbc3.roc(bted2, btec2), '\n')

	print()

	print('bgbc4', '\n')
	bgbc4 = GaussBayesClassify(btrd2, btrc2, equal=1, diagonal=1, wts=None, reg=np.asarray([1,3,4,6]))
	print(bgbc4, '\n')
	print(bgbc4.predict(bted2), '\n')
	print(bgbc4.predict_soft(bted2), '\n')
	print(bgbc4.auc(bted2, btec2), '\n')
	print(bgbc4.err(bted2, btec2), '\n')
	print(bgbc4.roc(bted2, btec2), '\n')

	print()

	print('ibgbc2', '\n')
	ibgbc2 = GaussBayesClassify(bted2, btec2, equal=1, diagonal=1, wts=None, reg=0)
	print(ibgbc2, '\n')
	print(ibgbc2.predict(btrd2), '\n')
	print(ibgbc2.predict_soft(btrd2), '\n')
	print(ibgbc2.auc(btrd2, btrc2), '\n')
	print(ibgbc2.err(btrd2, btrc2), '\n')
	print(ibgbc2.roc(btrd2, btrc2), '\n')


################################################################################
################################################################################
################################################################################
