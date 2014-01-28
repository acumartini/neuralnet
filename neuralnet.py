# title : neuralnet.py
# author : Adam Martini
# start_date : 1-7-14
#
# description : A vectorized implementation of an artificial neural network learning algorithm written 
#	in python using numpy. The final implementation of this project will support a variable number  of 
#	hidden layers (of variable size), multiclass classification, advanced optimized methods using scipy 
#	(BFGS, CG), code optimizations using LLVM inferfaces, and options for unsupervised training of DBN's.
#
# usage : python neuralnet.py <learning_rate> <regularization> <maxiter>
#	<batch_size> <hidden_layer_sizes>
#	(note: hidden layer sizes are separted by commas i.e., 10.10.10)
#
# python_version  : 3.3.3
#==============================================================================

import sys
import math
import csv
import numpy as np
# np.seterr(all='raise')
import mlutils as mlu

class NeuralNetClassifier():
	"""
	This class is responsible for all neural network classifier operations.  These operations include
	building a model from training data, class prediction for testing data, and printing the model.
	"""
	def __init__(self, units, lmbda, alpha, maxiter, batch_size):
		# user defined parameters
		self.units = units # the number of units in each hidden layer
		self.L = len(units) # the total number of layers including input and output layers
		self.alpha = float(alpha) # learning rate for gradient descent
		self.lmbda = float(lmbda) # regularization term
		self.maxiter = int(maxiter) # the maximum number of iterations through the data before stopping
		self.batch_size = int(batch_size) # for batch updates during gradient descent

		# internal parameters
		self.gtol = 0.00001 # convergence measure
		self.init_epsilon = 0.0001 # for random initialization of theta values
		self.threshold = 0.5 # the class prediction threshold

		# build network architecture by computing theta layer sizes and shapes
		self.sizes = [] # store theta layer divisions in the flattened theta array
		self.shapes = [] # store theta layer shapes for reshaping
		for i in range(len(self.units)-1):
			j_ = self.units[i+1]
			j = self.units[i]
			self.sizes.append(j_ * (j + 1))
			self.shapes.append((j_, j + 1))

		# randomly initialize weights for flattened theta array
		self.theta = np.random.rand(sum(self.sizes)) * (2 * self.init_epsilon) - self.init_epsilon

	def __str__(self):
		return "<Neural Network Classifier Instance: units=" + str(self.units) + ">\n"

	# =====================================================================================
	# Optimization

	def fit(self, X, y):
		"""
		This function optimizes the parameters for the neural network classification model from 
		training data using gradient descent.
		@post: parameters(theta) optimized by gradient descent using a cost and its derivative.
		"""
		self.n = X.shape[1] # the number of features
		self.m = X.shape[0] # the number of instances
		self.k = self.units[-1] # the numer of output units

		return self.minimize(self.cost, self.theta, X, y, self.jac, self.gtol)

		# # scipy advanced optimization
		# print("Performing advanced optimization using scipy.")
		# import scipy.optimize as opti
		# self.theta = opti.fmin_bfgs(self.cost, self.theta, fprime=self.jac, args=(X_, y_), 
		# 							maxiter=10, disp=True)

	def minimize(self, cost, theta, X, y, jac, gtol):
		costs = [] # store cost for plotting
		mags = [] # store gradient magnitudes for plotting

		# iterate through the data at most maxiter times, updating the theta for each feature
		# also stop iterating if error is less than epsilon (convergence tolerance constant)
		print("iter | batch | magnitude of the gradient")
		for iteration in range(self.maxiter):
			mags_tmp = [] # tmp magnitudes to average for iteration output

			# iterate through batches
			# t = time.time()
			batch_count = 0
			for X_, y_ in self.mini_batch(X, y):
				# compute the cost
				costs.append(self.cost(self.theta, X_, y_))

				# compute the gradient for current batch
				D = self.jac(self.theta, X_, y_)

				# perform gradient checking
				# print
				# grad_approx = self.estimate_gradient(X_, y_)
				# for i, ga in enumerate(grad_approx):
				# 	print i, D[i], ga

				# update theta parameters
				self.theta -= self.alpha * D

				# calculate the magnitude of the gradient and check for convergence
				mag = np.linalg.norm(D)
				mags.append(mag)
				mags_tmp.append(mag)
				if gtol > mag:
					break
				
				# print("iteration", iteration, ":", batch_count, ":", mag)
				batch_count += 1
			print("iteration", iteration, ":", np.mean(mags_tmp))
		return costs, mags

	def mini_batch(self, X, y):
		b = self.batch_size # var to clean up code

		if b == -1 or b >= self.m: # batch process by default
			yield (X, y)
		else:
			# test if the batch size requires a remainder yield
			size = float(self.m) / b
			final_batch = True if (size % 1 > 0) else False

			# iterate dataset and yield batches for gradient descent processing
			i = 0 # instance index
			for j in range(int(math.floor(size))):
				X_, y_ = X[i:i + b], y[i:i + b]
				yield (X_, y_)
				i += b

			if final_batch: # yield the remaining instances
				X_, y_ = X[i:], y[i:]
				yield (X_, y_)

	# =====================================================================================
	# Cost Function and Gradient Computation

	def cost(self, theta, X, y):
		# compute the cost function J(theta) using the regularization term lmbda
		m = X.shape[0]
		theta_sum = 0
		thetas = self.unpack_parameters(theta)
		for theta_j in thetas:
			theta_sum += (theta_j[:,1:] ** 2).sum()
		reg = (self.lmbda / (2 * m)) * theta_sum

		cost_sum = 0
		for i, x in enumerate(X):
			a, h_x = self.forward_prop(x, thetas)
			for k in range(self.k):
				cost_sum += (np.dot(y[i][k], np.log(h_x[k])) + np.dot((1 - y[i][k]), np.log(1 - h_x[k]))).sum()
		return ((- 1.0 / m) * cost_sum) + reg

	def jac(self, theta, X, y):
		m = X.shape[0] # number of instances

		# set the delta accumulator for gradient descent to 0
		self.delta = np.zeros((theta.shape))

		# iterate through instances and accumulate deltas
		for i, x in enumerate(X):
			# get theta parameter arrays for each layer
			thetas = self.unpack_parameters(theta)

			# calculate the activation values
			a, h_x = self.forward_prop(x, thetas)

			# back propagate the error
			self.back_prop(x, y[i], a, thetas)
		
		# compute the partial derivative terms with regularization
		theta_reg = np.array(())
		for theta in thetas:
			theta_j = np.copy(theta)
			theta_j *= self.lmbda # regularize the entire parameter matrix
			# remove regularization from theta_0 parameters
			theta_j = np.hstack((np.zeros((theta_j.shape[0], 1)), theta_j[:,1:]))
			theta_reg = np.hstack((theta_reg, theta_j.flatten()))

		# normalize and regularize the delta accumulator to obtain the final gradient
		return (1.0 / m) * (self.delta + theta_reg)

	def forward_prop(self, x_, thetas=None):
		a = np.array(()) # store activation values for each hidden and output layer unit

		# iterate through each layer in the network computing and forward propagating activation values
		x = x_ # preserve original x
		if thetas is None:
			thetas = self.unpack_parameters(self.theta)
		for theta_j in thetas:
			x = np.hstack((1, x)) # add bias unit with value 1.0
			a_ = self.compute_activation(np.dot(theta_j, x)) # the current layer's activation values
			x = a_ # populate x with new "features" for next iteration of activation calcs
			a = np.hstack((a, a_)) # record current layer activation values
		return a, a_

	def back_prop(self, x, y, a_, thetas):
		a = self.unpack_activations(a_)
		
		d = [a[-1] - y] # delta_L
		# iterate through layer activation values in reverse order computing d
		for j in reversed(range(1, self.L - 1)):
			a_tmp = np.hstack((1, a[j-1]))
			d_tmp = (np.dot(thetas[j].T, d[0]) * (a_tmp * (1 - a_tmp)))[1:]
			d.insert(0, d_tmp)

		a.insert(0, x)
		delta = np.array(())
		for l in range(1, self.L):
			delta_l =  np.outer(d[l-1], np.hstack((1, a[l-1])))
			delta = np.hstack((delta, delta_l.flatten()))
		self.delta += delta

	def estimate_gradient(self, X, y):
		epsilon = .0001 # the the one-sided distance from the actual theta parameter value

		# compute the derivative estimate with respect to each theta parameter
		grad_approx = np.zeros(self.theta.shape)
		for i in range(len(self.theta)):
			# adjust the current theta parameter based on elpsilon
			theta_plus = np.copy(self.theta)
			theta_plus[i] += epsilon
			theta_minus = np.copy(self.theta)
			theta_minus[i] -= epsilon

			# compute the two-sided difference
			cost_plus = self.cost(theta_plus, X, y)
			cost_minus = self.cost(theta_minus, X, y)
			grad_approx[i] = (cost_plus - cost_minus) / (2 * epsilon)

		return grad_approx

	# =====================================================================================
	# Model Architecture Utilities

	def compute_activation(self, z):
		return np.divide(1.0 , (1 + np.exp(- z)))

	def unpack_parameters(self, param):
		params = []
		i = 0 # store flattened theta array index value from previous iteration
		for j,s in zip(self.sizes, self.shapes):
			params.append(param[i:i+j].reshape(s[0], s[1])) # get current layers theta matrix
			i += j # record the flattened array index for the end of current layer
		return params

	def unpack_activations(self, a_):
		a = []
		i = 0 # store flattened activation array index value from previous iteration
		for j in self.units[1:]:
			a.append(a_[i:i+j]) # append current activation layer values
			i += j
		return a

	# ======================================================================================
	# Prediction

	def get_proba(self, X):
		return 1.0 / (1 + np.exp(- np.dot(X, self.theta)))

	def predict_proba(self, X):
		"""
		Returns the set of classification probabilities based on the model theta.
		@parameters: X - array-like of shape = [n_samples, n_features]
		    		 The input samples.
		@returns: y_pred - list of shape = [n_samples]
				  The probabilities that the class label for each instance is 1 to standard output.
		"""
		proba = []
		for x in X:
			a, h_x = self.forward_prop(x)
			if self.k < 2:
				proba.append(h_x[0])
			else:
				proba.append(h_x)
		return proba

	def predict(self, X):
		"""
		Classifies a set of data instances X based on the set of trained feature theta.
		@parameters: X - array-like of shape = [n_samples, n_features]
		    		 The input samples.
		@returns: y_pred - list of shape = [n_samples]
				  The predicted class label for each instance.
		"""
		probas = self.predict_proba(X)
		if self.k < 2:
			y_pred = [proba > self.threshold for proba in probas]
		else:
			y_pred = [np.argmax(proba) for proba in probas]
		return np.array(y_pred)

	# ==================================================================================
	# Model Output

	# def print_model(self, features, model_file):
	# 	# wite the parameter values corresponding to each feature to the given model file
	# 	with open(model_file, 'w') as mf:
	# 		for i in range(self.n):
	# 			if i == 0:
	# 				mf.write('%f\n' % (self.theta[i]))
	# 			else:
	# 				mf.write('%s %f\n' % (features[i-1], self.theta[i]))


def main(train_file, test_file, alpha=0.01, lmbda=0, maxiter=100, batch_size=-1, units=None):
	"""
	Manages files and operations for the neural network model creation, training, and testing.
	@parameters: alpha - the learning rate for gradient descent
				 maxiter - the maximum number of iterations allowed for training
				 lmbda - the regularization term
				 units - a sequence of integers separated by '.' sunch that each integer
				 represents the numer of units in a sequence of hidden layers.
	"""
	### ============== Digits dataset from sklearn ===================== ###
	# To use, comment out  the mlu.load_csv calls below

	# from sklearn import datasets, svm, metrics
	# # The digits dataset
	# digits = datasets.load_digits()
	# n_samples = len(digits.images)
	# data = digits.images.reshape((n_samples, -1))
	# X_train, y_train = data[:n_samples / 2], digits.target[:n_samples / 2]
	# X_test, y_test = data[n_samples / 2:], digits.target[n_samples / 2:]
	### ================================================================== ###

	# open and load csv files
	X_train, y_train = mlu.load_csv(train_file, True) # load and shuffle training set
	X_test, y_test = mlu.load_csv(test_file)

	# perform feature scaling
	X_train = mlu.scale_features(X_train, 0.0, 1.0)
	X_test = mlu.scale_features(X_test, 0.0, 1.0)
	# X_train = mlu.mean_normalize(X_train, True)
	# X_test = mlu.mean_normalize(X_test, True)

	# get units list
	input_units = int(X_train.shape[1])
	units_ = [input_units]
	if units is None:
		units_.extend([2 * input_units])
	else:
		units_.extend([int(u) for u in units.split('.')])

	# calculate the number of output units
	train_clss = np.unique(y_train) # get the unique elements of the labels array
	test_clss = np.unique(y_test)
	train_clss.sort()
	test_clss.sort()
	if not np.array_equal(train_clss, test_clss): # verify that training and testing set labels match
		print("Warning: Training and testing set labels do not agree.")
	
	# record the number of output units
	num_clss = len(train_clss)
	if num_clss == 2:
		units_.append(1)
	else:
		units_.append(num_clss) # record the number of output units

		# format dataset labels to multiclass classification arrays
		y_train = mlu.multiclass_format(y_train, num_clss)
		y_test_ = mlu.multiclass_format(y_test, num_clss)

	from sklearn import datasets, svm, metrics
	# The digits dataset
	digits = datasets.load_digits()

	# create the neural network classifier using the training data
	NNC = NeuralNetClassifier(units_, lmbda, alpha, maxiter, batch_size)
	print("\nCreated a neural network classifier =", NNC)

	# fit the model to the loaded training data
	# print("X_train.shape", X_train.shape)
	print("Fitting the training data...\n")
	costs, mags = NNC.fit(X_train, y_train)

	# predict the results for the test data
	print("Generating probability prediction for the test data...\n")
	y_pred = NNC.predict(X_test)

	### print the classification results ###
	print("The probabilities for each instance in the test set are:\n")
	for prob in NNC.predict_proba(X_test):
		print(prob)
	# print simple precision metric to the console
	print('Accuracy:  ' + str(mlu.compute_accuracy(y_test, y_pred)))
	
	# write the model to the model file
	# NNC.print_model(features, model_file)

	# plot convergence 
	import pylab
	pylab.xlabel('Iteration')
	pylab.ylabel('Cost')
	pylab.title('Costs')
	pylab.plot(range(len(costs)), costs)
	pylab.show()
	pylab.xlabel('Iteration')
	pylab.ylabel('Gradient')
	pylab.title('Gradients')
	pylab.plot(range(len(mags)), mags)
	pylab.show()


if __name__ == '__main__':
	"""
	The main function is called when neuralnet.py is run from the command line with arguments.
	"""
	args = sys.argv[1:] # get arguments from the command line
	main( *args )