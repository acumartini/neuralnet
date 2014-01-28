# Adam Martini
# pca.py
# A vectorized implementation of principal component analysis using numpy.
# Implementation in progress...

import numpy as np

class PCA():
	def __init__(self, min_retained=0.99, k=None):
		self.k = k # the number of principal components
		self.min_retained = min_retained # the percentage of variance retained

	def __str__(self):
		return "<Principal Component Analysis Instance>"

	def fit(self, X):
		self.sigma = np.cov(X)
		self.U, s, V = np.linalg.svg(self.sigma)

		# compute the number of components
		if self.k is None:
			d = s.sum() # denominator of variance equation

			for k in range(X.shape[1]):
				# compute the numerator
				n_sum = 0
				for i in range(k):
					n_sum += s[i,i]

				# compute variance retained
				retained = d / n_sum

				# return the number of components that satisfies min_variance
				if retianed >= self.min_retained:
					self.k = k + 1
					break

	def transform(self, X):
		U_reduce = self.U[:,:self.k]
		X_ = np.empty(X.shape[0], self.k)
		for i, x in enumerate(X):
			X_[i] = np.dot(self.U_reduce.T, x)
		return X_

	# def transform_inverse(self, X):
	# 	X_ = np.empty_like(X)
	# 	for i, x in enumerate(X):
	# 		X_[i] = np.dot(self.U_reduce.T, x)
	# 	return X_
