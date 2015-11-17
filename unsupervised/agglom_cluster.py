################################################################################
## IMPORTS #####################################################################
################################################################################


import data
import matplotlib.pyplot as plt
import numpy as np
import random

from numpy import asarray as arr
from numpy import asmatrix as mat
from numpy import atleast_2d as twod


################################################################################
################################################################################
################################################################################


################################################################################
## AGGLOMERATIVE ###############################################################
################################################################################


def agglom_cluster(X, n_clust, method='means', join=None):
	"""
	Perform hierarchical agglomerative 

	Parameters
	----------
	X : numpy array
	n_clust : 
	method : str
	join : 

	Returns
	-------
	"""
	m,n = twod(X).shape					# get data size
	D = np.zeros(__idx(m, m, m) + 1)	# store pairwise distances b/w clusters
	z = arr(range(m))					# assignments of data
	num = np.ones(m)					# number of data in each cluster
	mu = arr(X)							# centroid of each cluster
	method = method.lower()

	if type(join) == type(None):		# if join not precomputed
		join = np.zeros((m - 1, 3))		# keep track of join sequence
		# use standard Euclidean distance
		dist = lambda a,b: np.sum(np.power(a - b, 2))
		for i in range(m):				# compute initial distances
			for j in range(i + 1, m):
				D[__idx(i, j, m)] = dist(X[i,:], X[j,:])

	opn = np.ones(m)
	val = np.min(D)						# find first join (closest cluster pair)
	k = np.where(D == val)[0][0]
	print('D[622]')
	print(D[622])
	
	for c in range(m - 1):
		i,j = __ij(k, m)
		print('joining', i, '&', j)
		join[c,:] = arr([i, j, val])

		# centroid of new cluster
		mu_new = (num[i] * mu[i,:] + num[j] * mu[j,:]) / (num[i] + num[j])

		# compute new distances to cluster i
		for jj in np.where(opn)[0]:
			if jj in [i, j]:
				continue
			
			if method == 'min':
				pass
			elif method == 'max':
				pass
			elif method == 'means':
				D[__idx(i, jj, m)] = dist(mu_new, mu[jj,:])
			elif method == 'average':
				pass

		opn[j] = 0						# close cluster j (fold into i)
		num[i] = num[i] + num[j]		# update total membership in cluster i to include j
		mu[i,:] = mu_new				# update centroid list

		# remove cluster j from consideration as min
		for ii in range(m):
			if ii != j:
				D[__idx(ii, j, m)] = np.inf

		val,k = np.min(D), np.argmin(D)	# find next smallext pair
		print('val =', val, 'at k =', k, '| D[622] =', D[622])

	for c in range(m - n_clust):
		z[z == join[c,1]] = join[c,0]

	uniq = np.unique(z)
	for c in range(len(uniq)):
		z[z == uniq[c]] = c

	return z


def __idx(i, j, m):
	return (m * i - (i + 1) * i / 2 + (j + 1) - (i + 1)) - 1


def __ij(k, m):
	i = 1
	while k + 1 > m - i:
		k = k - (m - i)
		i += 1
	j = k + i
	return i - 1,j


################################################################################
################################################################################
################################################################################


################################################################################
## MAIN ########################################################################
################################################################################


if __name__ == '__main__':

	X,Y = data.load_data_from_csv('../classifier-data.csv', 4, float)
	X,Y = arr(X), arr(Y)

	z = agglom_cluster(X, 5)
	print('z')
	print(z)


################################################################################
################################################################################
################################################################################
