# Adam Martini
# pca.py
# A vectorized implementation of principal component analysis using numpy.

import numpy as np
import sys
import mlutils as mlu

class PCA():
	def __init__(self, k=None, min_retained=0.99):
		self.k = 50 #k # the number of principal components
		self.min_retained = min_retained # the percentage of variance retained

	def __str__(self):
		return "<Principal Component Analysis Instance>"

	def fit(self, X):
		# print "X", X.shape
		self.sigma = np.cov(X, rowvar=0)
		# self.sigma = np.cov(X)
		# print "sigma", self.sigma.shape
		self.U, s, V = np.linalg.svd(self.sigma)

		# print self.U.shape
		# print s.shape
		# print V.shape
		# for v in V:
		# 	print v
		# raise

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
		U_reduce = self.U[:,:self.k]
		X_ = np.empty((X.shape[0], self.k))
		for i, x in enumerate(X):
			# print U_reduce.T.shape
			# print x.shape
			X_[i] = np.dot(U_reduce.T, x)
		return X_

	# def transform_inverse(self, X):
	# 	X_ = np.empty_like(X)
	# 	for i, x in enumerate(X):
	# 		X_[i] = np.dot(self.U_reduce.T, x)
	# 	return X_

def main(*data_files):
	"""
	Manages files and operations for the neural network model creation, training, and testing.
	@parameters: *data_files - a list of paths to files to compress.  It is assumed that each
				 file has the same features.
	"""
	### ============== Digits dataset from sklearn ===================== ###
	# To use, comment out the mlu.load_csv() calls below

	# from sklearn import datasets, svm, metrics
	# # The digits dataset
	# digits = datasets.load_digits()
	# n_samples = len(digits.images)
	# data = digits.images.reshape((n_samples, -1))
	# X_train, y_train = data[:n_samples / 2], digits.target[:n_samples / 2]
	# X_test, y_test = data[n_samples / 2:], digits.target[n_samples / 2:]
	### ================================================================== ###	

	# open and load each dataset
	data = None
	indices = [] # store indices to split data after compression
	labels = [] # store the labels for each dataset

	j = 0 # keep track of last index
	for data_file in data_files:
		# load data and append to processing set
		X, y = mlu.load_csv(data_file) # load and shuffle training set
		print X.shape
		print y.shape
		if data is None:
			data = X
		else:
			print "VSTACK"
			data = np.vstack((data, X))
		# store reconstruction data
		indices.append(X.shape[0] + j)
		j = X.shape[0]
		labels.append(y)
		s = ''
		for x in X[0]:
			s += str(x) + ' '
		print s
		print "y", y[0]
		print "data.sahpe", data.shape
	print indices
	print

	# # create the neural network classifier using the training data
	# compressor = PCA()
	# print("\nCreated a principal component alalysis object =", compressor)

	# # fit the model to the loaded training data
	# # print("X_train.shape", X_train.shape)
	# print("Fitting the model to the dataset...\n")
	# compressor.fit(data)

	# # transform the data retaining 99% of tge variance
	# data = compressor.transform(data)
	
	# print("The final dataset...")
	# print("data.shape: ", data.shape)
	# # for d in data:
	# # 	print d
	 	
	# reconstruct datasets
	data_final = []
	j = 0
	print "data.shape", data.shape
	for i, y in zip(indices, labels):
		print "y.shape", y.shape
		print "i", i, "j", j
		d_tmp = np.hstack((data[j:i,:], y))
		j = i
		print "d_tmp.shape", d_tmp.shape
		s = ''
		for d in d_tmp[0]:
			s += str(d) + ' '
		print s
		print "y", y[0]
		data_final.append(d_tmp)
	 	
	# save data to disk
	for dataset, filename in zip(data_final, data_files):
		new_name = filename.replace('.csv', '.hdf')
		mlu.saveh(dataset, new_name)	

if __name__ == '__main__':
	"""
	The main function is called when neuralnet.py is run from the command line with arguments.
	"""
	args = sys.argv[1:] # get arguments from the command line
	main( *args )
