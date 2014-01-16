# Adam Martini
# Utilities for data preprocessing and classification evaluation.
# mlutils.py

import numpy as np

def load_csv(data):
	"""
	Loads the csv files into numpy arrays.
	@parameters: data The data file in csv format to be loaded
				 features True => the data file includes features as the first row 
	@returns: feature_names - None if features == False, else a list of feature names
			  X - numpy array of data instances with dtype=float
			  y - numpy array of labels
	"""
	print "Loading data from", data
	X = np.loadtxt(data, delimiter=",", dtype='float')
	y = X[:,-1:] # get only the labels
	# y = y.flatten() # make the single column 1 dimensional
	X = X[:,:-1] # remove the labels column from the data array

	return X, y

def scale_features(X, new_min, new_max):
	# scales all features in dataset X to values between new_min and new_max
	X_min, X_max = X.min(0), X.max(0)
	return (((X - X_min) / (X_max - X_min)) * (new_max - new_min + 0.000001)) + new_min

def multiclass_format(y, c):
	"""
	Formats dataset labels y to a vector representation for multiclass classification.
	i.e., If there are 3 classes {0,1,2}, then all instances of 0 are transformed to
	[1,0,0], 1''s are tranformed to [0,1,0], and 2's become [0,0,1]
	"""
	y_ = np.zeros(shape=(len(y), c));
	for i, lable in enumerate(y):
		y_[i][int(lable)] = 1.0
	return y_

def compute_accuracy(y_test, y_pred):
	"""
	@returns: The precision of the classifier, (correct labels / instance count)
	"""
	correct = 0
	for i, pred in enumerate(y_pred):
		if int(pred) == y_test[i]:
			correct += 1
	return float(correct) / y_test.size