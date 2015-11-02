################################################################################
## IMPORTS #####################################################################
################################################################################


import csv
import math
import numpy as np

from classify import Classify
from numpy import asarray as arr
from numpy import asmatrix as mat
from utils.data import load_data_from_csv, filter_data
from utils.test import test_randomly


################################################################################
################################################################################
################################################################################


################################################################################
## KNNCLASSIFY #################################################################
################################################################################


class KNNClassify(Classify):

	def __init__(self, X=None, Y=None, K=1, alpha=0):
		"""
		Constructor for KNNClassifier.  

		Parameters
		----------
		X : N x M numpy array 
			N = number of training instances; M = number of features.  
		Y : 1 x N numpy array 
			Contains class labels that correspond to instances in X.
		K : int 
			Sets the number of neighbors to used for predictions.
		alpha : scalar (int or float) 
			Weighted average coefficient (Gaussian weighting; alpha = 0 -> simple average).
		"""
		self.K = K
		self.X_train = []
		self.Y_train = []
		self.classes = []
		self.alpha = alpha

		if type(X) == np.ndarray and type(Y) == np.ndarray:
			self.train(X, Y)


	def __repr__(self):
		str_rep = 'KNNClassifier, {} classes, K={}{}'.format(
			len(self.classes), self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
			if self.alpha else '')
		return str_rep


	def __str__(self):
		str_rep = 'KNNClassifier, {} classes, K={}{}'.format(
			len(self.classes), self.K, ', weighted (alpha=' + str(self.alpha) + ')' 
			if self.alpha else '')
		return str_rep


## CORE METHODS ################################################################
			

	def train(self, X, Y):
		"""
		This method "trains" the KNNClassifier: it stores the input data and 
		determines the number of possible classes of data.  Refer to constructor
		doc string for descriptions of X and Y.
		"""
		self.X_train = np.asarray(X)
		self.Y_train = np.asarray(Y)
		self.classes = list(np.unique(Y))


	def predict_soft(self, X):
		"""
		This method makes a "soft" nearest-neighbor prediction on test data.

		Parameters
		----------
		X : N x M numpy array 
			N = number of testing instances; M = number of features.  
		"""
		tr_r,tr_c = np.asmatrix(self.X_train).shape					# get size of training data
		te_r,te_c = np.asmatrix(X).shape							# get size of test data
		
		num_classes = len(self.classes)
		prob = np.zeros((te_r,num_classes))							# allocate memory for class probabilities
		K = min(self.K, tr_r)										# can't have more than neighbors than there are training data points
		for i in range(te_r):										# for each test example...
			# ...compute sum of squared differences...
			dist = np.sum(np.power(self.X_train - np.asmatrix(X)[i,:], 2), axis=1)
			# ...find nearest neighbors over training data and keep nearest K data points
			sorted_dist = np.sort(dist, axis=0)[0:K]				
			indices = np.argsort(dist, axis=0)[0:K]				
			wts = np.exp(-self.alpha * sorted_dist)
			count = []
			for c in range(len(self.classes)):
				# total weight of instances of that classes
				count.append(np.sum(wts[self.Y_train[indices] == self.classes[c]]))
			count = np.asarray(count)
			prob[i,:] = np.divide(count, np.sum(count))				# save (soft) results
		return prob


	def predict(self, X):
		"""
		This method makes a nearest neighbor prediction on test data.
		Refer to the predict_soft doc string for a description of X.
		"""
		tr_r,tr_c = np.asmatrix(self.X_train).shape					# get size of training data
		te_r,te_c = np.asmatrix(X).shape							# get size of test data
		assert te_c == tr_c, 'Training and prediction data must have same number of features'
		
		num_classes = len(self.classes)
		Y_te = np.tile(self.Y_train[0], (te_r, 1))					# make Y_te same data type as Y_train
		K = min(self.K, tr_r)										# can't have more neighbors than there are training data points
		for i in range(te_r):										# for each test example...
			# ...compute sum of squared differences...
			dist = np.sum(np.power(self.X_train - np.asmatrix(X)[i,:], 2), axis=1)
			# ...find neares neighbors over training data and keep nearest K data points
			sorted_dist = np.sort(dist, axis=0)[0:K]
			indices = np.argsort(dist, axis=0)[0:K]
			wts = np.exp(-self.alpha * sorted_dist)
			count = []
			for c in range(len(self.classes)):
				# total weight of instances of that classes
				count.append(np.sum(wts[self.Y_train[indices] == self.classes[c]]))
			count = np.asarray(count)
			c_max = np.argmax(count)								# find largest count...
			Y_te[i] = self.classes[c_max]							# ...and save results
		return Y_te


## MUTATORS ####################################################################


	def set_alpha(self, alpha):
		"""
		Set weight parameter.  

		Parameters
		----------
		alpha : scalar (int or float)
		"""
		if type(alpha) not in [int, float]:
			raise TypeError('alpha must be of type int or float')
		self.alpha = alpha

	
	def set_classes(self, classes):
		"""
		Set classes.  

		Parameters
		----------
		classes : interable of class tags
		"""
		try:
			iter(classes)
			self.classes = classes
		except TypeError:
			raise TypeError('classes must be iterable')


	def set_K(self, K):
		"""
		Set K. 

		Parameters
		----------
		K : int
		"""
		if type(K) not in [int, float]:
			raise TypeError('K must be of type int or float')
		self.K = int(K)


## INSPECTORS ##################################################################


	def get_classes(self):
		return self.classes


	def get_K(self):
		return self.K


################################################################################
################################################################################
################################################################################


################################################################################
## CUSTOM EXCEPTIONS ###########################################################
################################################################################


class DifferentDimensionsError(Exception):
	pass


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

## RANDOM TESTING ##############################################################

	data,classes = load_data_from_csv('../classifier-data.csv', 4, float)
	data,classes = arr(data), arr(classes)

	bd1,bc1 = filter_data(data, classes, lambda x,y: y == 2)
	bd1,bc1 = np.array(bd1), np.array(bc1)

	def test(trd, trc, ted, tec):
		print('knn', '\n')
		knn = KNNClassify(trd, trc)
		print(knn, '\n')
#		print(knn.predict(ted), '\n')
#		print(knn.predict_soft(ted), '\n')
#		print(knn.confusion(ted, tec), '\n')
		print(knn.auc(ted, tec), '\n')
		print(knn.roc(ted, tec), '\n')
		err = knn.err(ted, tec)
		print(err, '\n')
		return err

	avg_err = test_randomly(bd1, bc1, 0.8, test)

	print('avg_err')
	print(avg_err)

#### DETERMINISTIC TESTING #######################################################
#
#	data = [[float(val) for val in row[:-1]] for row in csv.reader(open('../classifier-data.csv'))]
#	trd = np.asarray(data[0:40] + data[50:90] + data[100:140])
#	ted = np.asarray(data[40:50] + data[90:100] + data[140:150])
#	classes = [float(row[-1].lower()) for row in csv.reader(open('../classifier-data.csv'))]
#	trc = np.asarray(classes[0:40] + classes[50:90] + classes[100:140])
#	tec = np.asarray(classes[40:50] + classes[90:100] + classes[140:150])
#
#	btrd = trd[0:80,:]
#	bted = ted[0:20,:]
#	btrc = trc[0:80]
#	btec = tec[0:20]
#
#	btrd2 = trd[40:120,:]
#	bted2 = ted[10:30,:]
#	btrc2 = trc[40:120]
#	btec2 = tec[10:30]
#
#	print('bknn', '\n')
#	bknn = KNNClassify(btrd, btrc, K=5, alpha=3.2)
#	print(bknn, '\n')
#	print(bknn.predict(bted), '\n')
#	print(bknn.predict_soft(bted), '\n')
#	print(bknn.auc(bted, btec), '\n')
#	print(bknn.err(bted, btec), '\n')
#	print(bknn.roc(bted, btec), '\n')
#
#	print()
#
#	print('ibknn', '\n')
#	ibknn = KNNClassify(bted, btec, K=3, alpha=1.5)
#	print(ibknn, '\n')
#	print(ibknn.predict(btrd), '\n')
#	print(ibknn.predict_soft(btrd), '\n')
#	print(ibknn.auc(btrd, btrc), '\n')
#	print(ibknn.err(btrd, btrc), '\n')
#	print(ibknn.roc(btrd, btrc), '\n')
#
#	print()
#
#	print('bknn2', '\n')
#	bknn2 = KNNClassify(btrd2, btrc2, K=2, alpha=2.3)
#	print(bknn2, '\n')
#	print(bknn2.predict(bted2), '\n')
#	print(bknn2.predict_soft(bted2), '\n')
#	print(bknn2.auc(bted2, btec2), '\n')
#	print(bknn2.err(bted2, btec2), '\n')
#	print(bknn2.roc(bted2, btec2), '\n')
#
#	print()
#
#	print('ibknn2', '\n')
#	ibknn2 = KNNClassify(bted2, btec2, K=3, alpha=1.5)
#	print(ibknn2, '\n')
#	print(ibknn2.predict(btrd2), '\n')
#	print(ibknn2.predict_soft(btrd2), '\n')
#	print(ibknn2.auc(btrd2, btrc2), '\n')
#	print(ibknn2.err(btrd2, btrc2), '\n')
#	print(ibknn2.roc(btrd2, btrc2), '\n')
#
#
################################################################################
################################################################################
################################################################################



