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
## CLUSTERING ##################################################################
################################################################################


## AGGLOMERATIVE ###############################################################


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
	z = twod(arr(range(m))).T			# assignments of data
	num = np.ones(m)
	mu = arr(X)
	method = method.lower()

	if type(join) == type(None):		# if join not precomputed
		join = np.zeros((m - 1, 3))		# keep track of join sequence
		# use standard Euclidean distance
		dist = lambda a,b: np.sum(np.power(a - b, 2))
		for i in range(m):				# compute initial distances
			for j in range(i + 1, m):
				D[__idx(i, j, m)] = dist(X[i,:], X[j,:])

	opn = np.ones(m)
	val,k = np.min(D), np.argmin(D)		# find first join (closest cluster pair)
	
	for c in range(m - 1):
		i,j = __ij(k, m)
		print('k')
		print(k)
		print('joining', i, '&', j)
		join[c,:] = arr([i, j, val])
		print('D[622]')
		print(D[622])

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

		val,k = D.min(), D.argmin()		# find next smallext pair

		print('D[622]')
		print(D[622])
		print('val =', val, 'at k =', k)

	for c in range(m - n_clust):
		z[z == join[c,1]] = join[c,0]

	print('join')
	print(join)
	print(np.unique(join[:,0]).shape)
	print(np.unique(join[:,1]).shape)

	uniq = np.unique(z)
	print('uniq')
	print(uniq)
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


## EM ##########################################################################


def em_cluster(X, K, init='random', max_iter=100, tol=1e-6, do_plot=False):
	"""
	Perform Gaussian mixture EM clustering on data X.

	Parameters
	----------
	X : numpy array
		N x M array containing data to be clustered.
	K : int
		Number of clusters.
	init : str or array (optional)
		Either a K x N numpy array containing initial clusters, or
		one of the following strings that specifies a cluster init
		method: 'random' (K random data points (uniformly) as clusters),
		'farthest' (choose cluster 1 uniformly, then the point farthest
		from all cluster so far, etc.), or 'k++' (choose cluster 1 
		uniformly, then points randomly proportional to distance from
		current clusters).
	max_iter : int
	tol : scalar

	Returns
	-------

	"""
	N,D = twod(X).shape					# get data size


	


## KMEANS ######################################################################


def kmeans(X, K, init='random', max_iter=100, do_plot=False, to_return='z'):
	"""
	Perform K-means clustering on data X.

	Parameters
	----------
	X : numpy array
		N x M array containing data to be clustered.
	K : int
		Number of clusters.
	init : str or array (optional)
		Either a K x N numpy array containing initial clusters, or
		one of the following strings that specifies a cluster init
		method: 'random' (K random data points (uniformly) as clusters),
		'farthest' (choose cluster 1 uniformly, then the point farthest
		from all cluster so far, etc.), or 'k++' (choose cluster 1 
		uniformly, then points randomly proportional to distance from
		current clusters).
	max_iter : int (optional)
		Maximum number of optimization iterations.
	do_plot : bool (optional)
		Plot 2D data?
	to_return : str

	Returns
	-------
	z : numpy array
		N x 1 array containing cluster numbers of data at indices
		in X.
	c : numpy array (optional)
		K x M array of cluster centers.
	sumd : scalar (optional)
		Sum of squared euclidean distances.
		

	TODO: test more
	"""
	n,d = twod(X).shape							# get data size

	if type(init) is str:
		init = init.lower()
		if init == 'random':
			pi = np.random.permutation(n)
			c = X[pi[0:K],:]
		elif init == 'farthest':
			c = __k_init(X, K, True)
		elif init == 'k++':
			c = __k_init(X, K, False)
		else:
			raise ValueError('KMeans.__init__: value for "init" ( ' + init +  ') is invalid')
	else:
		c = init

	return __optimize(X, n, K, c,  max_iter)



def __optimize(X, n, K, c, max_iter):
	iter = 1
	done = (iter > max_iter)
	sumd = np.inf
	sum_old = np.inf

	z = np.zeros((n,1))

	while not done:
		sumd = 0
		
		for i in range(n):
			# compute distances for each cluster center
			dists = np.sum(np.power((c - np.tile(X[i,:], (K,1))), 2), axis=1)
			val = np.min(dists, axis=0)							# assign datum i to nearest cluster
			z[i] = np.argmin(dists, axis=0) + 1
			sumd = sumd + val

		for j in range(1, K + 1):								# now update each cluster center j...
			if np.any(z == j):
				c[j - 1,:] = np.mean(X[(z == j).flatten(),:], 0)# ...to be the mean of the assigned data...
			else:
				c[j - 1,:] = X[np.floor(np.random.rand()),:]	# ...or random restart if no assigned data

		done = (iter > max_iter) or (sumd == sum_old)
		sum_old = sumd
		iter += 1

	return z
			

def __k_init(X, K, determ):
	"""
	Distance based initialization. Randomly choose a start point, then:
	if determ == True: choose point farthest from the clusters chosen so
	far, otherwise: randomly choose new points proportionally to their
	distance.

	Parameters
	----------
	X : numpy array
		See kmeans docstring.
	K : int
		See kmeans docstring.
	determ : bool
		See description.

	Returns
	-------
	c : numpy array
		K x M array of cluster centers.
	"""
	m,n = twod(X).shape
	clusters = np.zeros((K,n))
	clusters[0,:] = X[np.floor(np.random.rand() * m),:]			# take random point as first cluster
	dist = np.sum(np.power((X - np.ones((m,1)) * clusters[0,:]), 2), axis=1)

	for i in range(1,K):
		if determ:
			j = np.argmax(dist)									# choose farthest point...
		else:
			pr = np.cumsum(dist);								# ...or choose a random point by distance
			pr = pr / pr[-1]
			j = np.where(np.random.rand() < pr)[0][0]

		clusters[i,:] = X[j,:]									# update that cluster
		# update min distances
		new_dist = np.sum(np.power((X - np.ones((m,1)) * clusters[i,:]), 2), axis=1) 
		dist = np.minimum(dist, new_dist)

	return clusters
		

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