################################################################################
## IMPORTS #####################################################################
################################################################################


import math
import numpy as np


################################################################################
################################################################################
################################################################################


################################################################################
## BASECLASSIFY ################################################################
################################################################################


class BaseClassify:

	def __init__(self):
		"""
		Constructor for base class for several different classifier. 
		This class implements methods that generalize to different classifiers.
		"""
		self.classes = []


	def auc(self, X, Y):
		"""
		This method computes the area under the roc curve on the given test data.
		This method only works on binary classifiers. X should be an N x M
		numpy array of N data points with M features. Y should be a 1 x N numpy
		array of classes that refer to the data points in X.

		NOTE: This method generalizes to: bagged_classify, gauss_bayes_classify,
		knn_classify, linear_classify, logistic_classify, logistic_mse_classify,
		nnet_classify, tree_classify 
		"""
		if len(self.classes) > 2:
			raise ValueError('This method can only supports binary classification ')

		try:
			soft = self.predict_soft(X)[:,1]
		except IndexError:
			soft = self.predict(X)

		sorted_soft = np.sort(soft)
		indices = np.argsort(soft)
		same = np.append(np.asarray(sorted_soft[0:-1] == sorted_soft[1:]), 0)

		n = len(soft)
		rnk = self.__compute_ties(n, same)
		
		n0 = sum(Y == self.classes[0])
		n1 = sum(Y == self.classes[1])

		if n0 == 0 or n1 == 0:
			raise ValueError('Data of both class values not found')

		result = (np.sum(rnk[Y == self.classes[1]]) - n1 * (n1 + 1) / 2) / n1 / n0
		return result


	def confusion(self, X, Y):
		"""
		This method estimates the confusion matrix (Y x Y_hat) from test data.
		Refer to auc doc string for descriptions of X and Y.

		NOTE: This method generalizes to: gauss_bayes_classify, knn_classify,
		nnet_classify, tree_classify
		"""
		Y_hat = self.predict(X)
		num_classes = len(self.classes)
		indices = self.to_index(Y, self.classes) + num_classes * (self.to_index(Y_hat, self.classes) - 1)
		C = np.histogram(indices, np.asarray(range(1, num_classes**2 + 2)))[0]
		C = np.reshape(C, (num_classes, num_classes))
		return np.transpose(C)


	def err(self, X, Y):
		"""
		This method computes the error rate on test data.  X is an N x M
		numpy array of N data points with M features.  Y is a 1 x N numpy
		array of class values corresponding to the data points in X.

		NOTE: This method generalizes to: bagged_classify, gauss_bayes_classify,
		knn_classify, linear_classify, logistic_classify, logistic_mse_classify,
		nnet_classify, tree_classify 
		"""
		Y_hat = self.predict(X)
		Y_hat = np.transpose(Y_hat)
		return np.mean(Y_hat != Y)


	def predict(self, X):
		"""
		This is an abstract predict method that must exist in order to
		implement certain BaseClassify methods.
		"""
		pass


	def roc(self, X, Y):
		"""
		This method computes the "receiver operating characteristic" curve on
		test data.  This method only works for binary classifiers. Refer 
		to the auc doc string for descriptions of X and Y. Method returns
		[fpr, tpr, tnr]. Plot fpr and tpr to see the ROC curve. Plot tpr and
		tnr to see the sensitivity/specificity curve.

		NOTE: This method generalizes to: bagged_classify, gauss_bayes_classify,
		knn_classify, linear_classify, logistic_classify, logistic_mse_classify,
		nnet_classify, tree_classify 
		"""
		if len(self.classes) > 2:
			raise ValueError('This method can only supports binary classification ')

		try:
			soft = self.predict_soft(X)[:,1]
		except IndexError:
			soft = self.predict(X)

		n0 = sum(Y == self.classes[0])
		n1 = sum(Y == self.classes[1])

		if n0 == 0 or n1 == 0:
			raise ValueError('Data of both class values not found')

		sorted_soft = np.sort(soft)
		indices = np.argsort(soft, axis=0)
		Y = Y[indices]
		tpr = np.divide(np.cumsum(Y[::-1] == self.classes[1]), n1)
		fpr = np.divide(np.cumsum(Y[::-1] == self.classes[0]), n0)
		tnr = np.divide(np.cumsum(Y == self.classes[0]), n0)[::-1]

		same = np.append(np.asarray(sorted_soft[0:-1] == sorted_soft[1:]), 0)
		tpr = np.append([0], tpr[np.logical_not(same)])
		fpr = np.append([0], fpr[np.logical_not(same)])
		tnr = np.append([1], tnr[np.logical_not(same)])
		return [tpr, fpr, tnr]


## HELPERS #####################################################################


	def __compute_ties(self, n, same):
		rnk = list(range(n + 1))
		i = 0
		while i < n:
			if same[i]:
				start = i
				while same[i]:
					i += 1
				for j in range(start, i + 1):
					rnk[j] = float((i + 1 + start) / 2)
			i += 1
		return np.asarray(rnk[1:])


	def to_index(self, Y, values=None):
		values = values if values else list(np.unique(Y))
		m,n = np.asmatrix(Y).shape
		Y = Y.ravel() if m > n else Y
		m,n = np.asmatrix(Y).shape
		assert m == 1, 'Y must be discrete scalar'

		Y_ext = np.asarray([0 for i in range(n)])
		for i in range(len(values)):
			Y_ext[np.nonzero(Y == self.classes[i])[0]] = i
		return np.asarray([i + 1 for i in Y_ext])


################################################################################
################################################################################
################################################################################
