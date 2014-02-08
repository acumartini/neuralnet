# Adam Martini
# Utilities for data preprocessing/management and classification evaluation.
# mlutils.py

import numpy as np
import h5py
import csv

### DATA IO ###

def load_data(data, method, shuffle=False, split=True):
	if method == "csv":
		return load_csv(data, shuffle, split)
	elif method == "hdf":
		return loadh(data, shuffle, split)
	else:
		raise IOError("Data format not recognized.")

def save_data(data, method, path):
	if method == "csv":
		return save_csv(data, path)
	elif method == "hdf":
		return saveh(data, path)	
	else:
		raise IOError("Data format not recognized.")

def load_csv(data, shuffle=False, split=True):
	"""
	Loads the csv files into numpy arrays.
	@parameters: data The data file in csv format to be loaded
				 shuffle True => shuffle data instances to randomize
	@returns: X - numpy array of data instances with dtype=float
			  y - numpy array of labels
	"""
	print "Loading data from", data
	dset = np.loadtxt(data, delimiter=",", dtype='float32')
	return shuffle_split(dset, shuffle, split)

def save_csv(data, path):
	# write data to new csv file
	with open(path, 'wb') as csv_file:
		writer = csv.writer(csv_file, delimiter=',',quoting=csv.QUOTE_MINIMAL)
		writer.writerows(data)

def saveh(dset, path):
	"""
	Stores the numpy data in h5 format.
	@parameters: dset Dataset to store
				 path The path to file (including the file name) to save h5 file to
	"""
	f = h5py.File(path, 'w')
	f['dset'] = dset
	f.close()

def loadh(path, shuffle=False, split=True):
	"""
	Loads the h5 data into a numpy array.
	@parameters: path The path to file (including the file name) to load data from
				 shuffle True => shuffle data instances to randomize
	@returns: X - numpy array of data instances with dtype=float
			  y - numpy array of labels
	"""
	f = h5py.File(path,'r') 
	data = f.get('dset') 
	dset = np.array(data)
	return shuffle_split(dset, shuffle, split)

def shuffle_split(dset, shuffle, split):
	# randomize data
	if shuffle:
		dset = shuffle_data(dset)
	# split instances and labels
	if split:
		y = dset[:,-1:] # get only the labels
		X = dset[:,:-1] # remove the labels column from the data array
		dset = (X, y)
	return dset

def shuffle_data(data):
	# get a random list of indices
	rand_indices = np.arange(data.shape[0])
	np.random.shuffle(rand_indices)

	# build shuffled array
	data_ = np.zeros(data.shape)
	for i, index in enumerate(rand_indices):
		data_[i] = data[index]
	return data_

### DATA PREPROCESSING ###

def mean_normalize(X, std=False):
	# normalize the mean to 0 for each feature and scale based on max/min values or
	# the standard deviation according to parameter "std"
	d = X.std(0) if std else X.max(0) - X.min(0)
	return (X - X.mean(0)) / d

def scale_features(X, new_min, new_max):
	# scales all features in dataset X to values between new_min and new_max
	X_min, X_max = X.min(0), X.max(0)
	return (((X - X_min) / (X_max - X_min)) * (new_max - new_min + 0.000001)) + new_min

def multiclass_format(y, c):
	"""
	Formats dataset labels y to a vector representation for multiclass classification.
	i.e., If there are 3 classes {0,1,2}, then all instances of 0 are transformed to
	[1,0,0], 1''s are transformed to [0,1,0], and 2's become [0,0,1]
	"""
	if c == 2: # standard classification problem, formatting not required
		return y
	else:
		y_ = np.zeros(shape=(len(y), c));
		for i, lable in enumerate(y):
			y_[i][int(lable)] = 1.0
		return y_

### RESULT METRICS ###

def compute_accuracy(y, y_):
	"""
	@returns: The precision of the classifier, (correct labels / instance count)
	"""
	correct = 0
	for i, pred in enumerate(y_):
		if int(pred) == y[i]:
			correct += 1
	return float(correct) / y.size

def misclassification_error(y, y_):
	return 1 - compute_accuracy(y, y_)

def get_pos_precision(self, CM):
		Tn, Fp = CM[0]
		Fn, Tp = CM[1]
		if Tp + Fp == 0:
			return 0.0
		return float(Tp) / (Tp + Fp)

def get_pos_recall(self, CM):
	Tn, Fp = CM[0]
	Fn, Tp = CM[1]
	if Tp + Fn == 0:
		return 0.0
	return float(Tp) / (Tp + Fn)

def get_f_measure(self, P, R):
	return ((1 + self.beta**2) * P * R) / (((self.beta**2) * P) + R)

### Test Datasets ### 
# TODO: add test dataset loading functionality

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

### Plotting Utilities ###
# TODO: add plotting utility functionality
# plot convergence 
# import pylab
# pylab.xlabel('Iteration')
# pylab.ylabel('Cost')
# pylab.title('Costs')
# pylab.plot(range(len(costs)), costs)
# pylab.show()
# pylab.xlabel('Iteration')
# pylab.ylabel('Gradient')
# pylab.title('Gradients')
# pylab.plot(range(len(mags)), mags)
# pylab.show()