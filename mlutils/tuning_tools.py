# Adam Martini
# 
# Classifier tuning and plotting tools.
# 
# tuning_tools.py
# 2-9-14

import sys
from sets import Set
import cPickle
import numpy as np
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from mlutils import mlutils as mlu
from sklearn.base import clone

# plot formating variables
# TODO: Set plotting font to DejaVu Sans Mono
# mpl.font_manager.findfont('DejaVu Sans Mono', directory='~/Library/Fonts') # TODO: specify system path?
# sys_font = {'family' : 'DejaVu Sans Mono'}
font = {'family' : 'sans-serif',
		'style'  : 'normal',
		'weight' : 'semibold'}

def tune_features(clf, X_train, y_train, X_val, y_val, plot=False, index_rank=None, error_func=None):
	"""
	Returns the feature set that minimizes the error on the validation data.
	@parameters:
		index_rank - path to a pickled list of ordered indices
	"""
	# setup feature selection order index
	if index_rank is None:
		indices = range(X_train.shape[1] - 1)
	else:
		indices = cPickle.load(index_rank)

	# validate error function input
	if error_func is None:
		error_func = mlu.misclassification_error

	# train/test iterative storing results for plotting etc.
	error_train = [] # store training set errors
	error_val = [] # store validation set errors
	feats_best = None # stores the feature set tuple that minimizes the validation set error
	
	# create the initial training/validation feature arrays to stack onto
	X = X_train[:,indices[0]].reshape(X_train.shape[0], 1)
	X_ = X_val[:,indices[0]].reshape(X_val.shape[0], 1)
	i_0 = indices.pop(0)
	for e, i in enumerate(indices):
		# add the next feature if this is not the first loop
		if e != 0:
			X = np.hstack((X, X_train[:,i].reshape(X_train.shape[0], 1)))
			X_ = np.hstack((X_, X_val[:,i].reshape(X_val.shape[0], 1)))

		# create a deep copy of the original estimator
		clf_tmp = clone(clf)

		print "number of features:", e
		# train/test and store
		# training set
		clf_tmp.fit(X, y_train)
		y_pred = clf_tmp.predict(X)
		error_train.append(error_func(y_train, y_pred))

		# validation set
		y_pred = clf_tmp.predict(X_)
		error_tmp = error_func(y_val, y_pred)
		
		# check for best estimator
		if len(error_val) > 0 and error_tmp < min(error_val):
			feats_best = ([i_0] + indices[0:i], X, X_)
		error_val.append(error_tmp)

	# generate error plot
	if plot:
		fig, ax = plt.subplots(figsize=(12,9))
		ax.plot(error_val, label="Validation Set Error")
		ax.plot(error_train, label="Training Set Error")

		ax.set_xlabel('Number of Features', fontsize=18)
		ax.set_ylabel('Error', fontsize=18)

		title = ax.set_title('Feature Set Error Analysis', fontsize=22)
		title.set_y(1.03) # adjust space between title and graph

		ax.legend(loc='best')
		ax.grid(True)

		# plt.rc('font', **font)
		plt.savefig(("feature_set_error.png"))
		plt.show()

	return feats_best

def tune_regularization(clf, X, y, X_, y_, plot=False, error_func=None):
	# regularization terms to minimize over (inverse regularization C)
	C = [2**-15,2**-10, 2**-8, 2**-6, 2**-4, 2**-2, 
		 2**-1, 2**1, 2**2, 2**4, 2**6, 2**8, 2**10, 2**15]
	C.reverse()

	# validate error function input
	if error_func is None:
		error_func = mlu.misclassification_error

	# train/test iterative storing results for plotting etc.
	error_train = [] # store training set errors
	error_val = [] # store validation set errors
	C_best = None # stores the regularization term that minimizes validation set error

	for C_ in C:
		# create a deep copy of the original estimator
		clf_tmp = clone(clf)
		clf_tmp.set_params(C=C_)

		# train/test and store
		# training set
		clf_tmp.fit(X, y)
		y_pred = clf_tmp.predict(X)
		error_train.append(error_func(y, y_pred))

		# validation set
		y_pred = clf_tmp.predict(X_)
		error_tmp = error_func(y_, y_pred)
		
		# check for best estimator
		if len(error_val) > 0 and error_tmp < min(error_val):
			C_best = (C_)
		error_val.append(error_tmp)

	# generate error plot
	if plot:
		fig, ax = plt.subplots(figsize=(12,9))
		ax.plot(error_val, label="Validation Set Error")
		ax.plot(error_train, label="Training Set Error")
		
		ax.set_xlabel('Regularization Constant C', fontsize=18)
		ax.set_xticks(range(len(C)))
		ax.set_xticklabels([r'$2^{15}$', r'$2^{10}$', r'$2^8$', r'$2^6$', r'$2^4$', r'$2^2$',
						   r'$2$', r'$2^{-1}$', r'$2^{-2}$', r'$2^{-4}$', r'$2^{-6}$', 
						   r'$2^{-8}$', r'$2^{-10}$', r'$2^{-15}$'], fontsize=18)
		
		ax.set_ylabel('Error', fontsize=18)

		title = ax.set_title('Regularization Constant Error Analysis', fontsize=22)
		title.set_y(1.03) # adjust space between title and graph

		ax.legend(loc='best')
		ax.grid(True)

		# plt.rc('font', **font)
		plt.savefig(("regularization_error.png"))
		plt.show()

	return C_best

def instance_count_analysis(clf, X, y, X_, y_, plot=False, error_func=None):
	# validate error function input
	if error_func is None:
		error_func = mlu.misclassification_error

	# train/test iterative storing results for plotting etc.
	min_samples = len(X)
	error_train = [] # store training set errors
	error_val = [] # store validation set errors

	# get minimum instance size so that there are positive and negative examples
	unique_clss = np.unique(y)
	s = Set()
	min_count = 0
	while len(s) != len(unique_clss):
		s.add( y[min_count] )
		min_count += 1

	# build breakpoint indices array
	if len(X) >= min_samples: # the maximum number of data points is 100
		split = int(len(X) / min_samples)
		indices = []
		i = split
		while i + split < min_count:
			i += split
		indices.append(i)
		while i + split < len(X):
			indices.append(i + split)
			i += split
		indices.append(len(X))
	else:
		indices = range(min_count, len(X))

	for i in indices:
		# create a deep copy of the original estimator
		clf_tmp = clone(clf)

		# get the training instance dataset for this iteration
		X_tmp = X[0:i]
		y_tmp = y[0:i]

		# train/test and store
		# training set
		clf_tmp.fit(X_tmp, y_tmp)
		y_pred = clf_tmp.predict(X_tmp)
		error_train.append(error_func(y_tmp, y_pred))

		# validation set
		y_pred = clf_tmp.predict(X_)
		error_tmp = error_func(y_, y_pred)
		error_val.append(error_tmp)
		
	# generate error plot
	if plot:
		fig, ax = plt.subplots(figsize=(12,9))
		ax.plot(indices, error_val, label="Validation Set Error")
		ax.plot(indices, error_train, label="Training Set Error")
		
		ax.set_xlabel("Number of Instances", fontsize=18)
		
		ax.set_ylabel('Error', fontsize=18)

		title = ax.set_title('Instance Count Error Analysis', fontsize=22)
		title.set_y(1.03) # adjust space between title and graph

		ax.legend(loc='best')
		# ax.grid(True)

		# plt.rc('font', **font)
		plt.savefig(("instance_count_error.png"))
		plt.show()

def main(traindata, valdata, plot=False, index_rank=None):
	# load data
	X, y = mlu.load_csv(traindata, True)
	y = y.flatten()
	X_, y_ = mlu.load_csv(valdata, True)
	y_ = y_.flatten()

	# X = mlu.scale_features(X, -3.0, 3.0)
	# X_ = mlu.scale_features(X_, -3.0, 3.0)
	# X = mlu.mean_normalize(X)
	# X_ = mlu.mean_normalize(X_)

	# validate plot input
	plot = True if plot == "plot" else False

	# initialize classifier
	from sklearn.linear_model import LogisticRegression as LR
	clf = LR(tol=.00001, penalty='l2')

	# perform tuning steps
	print "\nPerforming feature set tuning.\n"
	indices, X, X_ = tune_features(clf, X, y, X_, y_, plot, index_rank)

	# perform regularization tuning with best feature set
	print "Performing regularization term tuning.\n"
	C = tune_regularization(clf, X, y, X_, y_, plot)

	# perform incremental instance error analysis
	print "Performing instance count analysis.\n"
	instance_count_analysis(clf, X, y, X_, y_, plot)

if __name__ == '__main__':
	"""
	The main function is called when tuning_tools.py is run from the command line with arguments.
	"""
	args = sys.argv[1:] # get arguments from the command line
	main( *args )

# Note: get most informative features from sklearn class
# def show_most_informative_features(vectorizer, clf, n=20):
    # c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    # top = zip(c_f[:n], c_f[:-(n+1):-1])
    # for (c1,f1),(c2,f2) in top:
    #     print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1,f1,c2,f2)
