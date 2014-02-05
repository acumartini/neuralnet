# Adam Martini
# pca.py
# A vectorized implementation of principal component analysis using numpy.
# Implementation in progress...

import numpy as np
import sys
import mlutils as mlu

class PCA():
	"""
	This class is responsible for all principal component analysis functionality.  The can choose to set
	the number of components, or that can be determined automatically based on the desired amount of
	retained variance.
	"""
	def __init__(self, k=None, min_retained=0.99):
		self.k = k # the number of principal components
		self.min_retained = min_retained # the percentage of variance retained

	def __str__(self):
		return "<Principal Component Analysis Instance>"

	def fit(self, X):
		"""
		Computed the principal component vectors for dimensionality reduction as well as the number
		of components needed to achieve the desired amount of retained variance (if k is not specified).
		"""
		self.sigma = np.cov(X, rowvar=0)
		self.U, s, V = np.linalg.svd(self.sigma)

		# compute the number of components
		if self.k is None:
			s_total = s.sum() # denominator of variance equation

			for k in range(X.shape[1]):
				# compute the numerator
				k_sum = s[:k].sum()

				# compute variance retained
				retained = k_sum / s_total

				# return the number of components that satisfies min_variance
				if retained >= self.min_retained:
					self.k = k + 1
					break

	def transform(self, X):
		"""
		Reduces the dimensionality of dataset X based on the model and number of components.
		@parameters X - the dataset to reduce
		@ returns - X_ - the dataset reduced to k components
		"""
		U_reduce = self.U[:,:self.k]
		X_ = np.empty((X.shape[0], self.k))
		for i, x in enumerate(X):
			X_[i] = np.dot(U_reduce.T, x)
		return X_

	# TODO
	# def transform_inverse(self, X):
	# 	X_ = np.empty_like(X)
	# 	for i, x in enumerate(X):
	# 		X_[i] = np.dot(self.U_reduce.T, x)
	# 	return X_

def main(*data_files):
	"""
	Manages files and operations for principal component analysis.
	@parameters: *data_files - a list of paths to files to process.  It is assumed that each
				 file has the same features.
	"""
	# open and load each dataset
	data = None
	indices = [] # store indices to split data after compression
	labels = [] # store the labels for each dataset

	j = 0 # keep track of last index
	for data_file in data_files:
		# load data and append to processing set
		X, y = mlu.load_csv(data_file) # load and shuffle training set
		if data is None:
			data = X
		else:
			data = np.vstack((data, X))
		# store reconstruction data
		indices.append(X.shape[0] + j)
		j = X.shape[0]
		labels.append(y)

	# use sklearn's implementation of PCA for now
	from sklearn.decomposition import PCA as PCAsk
	
	# fit dataset without component restriction
	pca = PCAsk()
	pca.fit(data)

	# determine the number of components required to retain 99% of the variance
	v_ratios = pca.explained_variance_ratio_
	k = 0
	variance = v_ratios[0]
	while variance < 0.99:
		k += 1
		variance += v_ratios[k]
	k += 1

	# fit dataset again with component restriction
	pca = PCAsk(n_components=k)
	pca.fit(data)

	# reduce dimensionality
	print("Performing data reduction...")
	data = pca.transform(data)

	# My PCA class still requires development and debugging
	# create the neural network classifier using the training data
	# compressor = PCA()
	# print("\nCreated a principal component analysis object =", compressor)

	# # fit the model to the loaded training data
	# # print("X_train.shape", X_train.shape)
	# print("Fitting the model to the dataset...\n")
	# compressor.fit(data)

	# # transform the data retaining 99% of the variance
	# data = compressor.transform(data)
	 	
	# reconstruct datasets
	data_final = []
	j = 0
	for i, y in zip(indices, labels):
		d_tmp = np.hstack((data[j:i,:], y))
		j = i
		data_final.append(d_tmp)
	 	
	# save data to disk
	for dataset, filename in zip(data_final, data_files):
		new_name = filename.replace('.csv', '.hdf')
		print("Storing reduced dataset: ", new_name)
		mlu.saveh(dataset, new_name)	

if __name__ == '__main__':
	"""
	The main function is called when neuralnet.py is run from the command line with arguments.
	"""
	args = sys.argv[1:] # get arguments from the command line
	main( *args )
