# title : neuralnet.py
# author : Adam Martini
# start_date : 1-7-14
#
# description : A vectorized implementation of an artificial neural network learning algorithm written 
#	in python using numpy. The final implementation of this project will support a variable number  of 
#	hidden layers (of variable size), multiclass classification, advanced optimized methods using scipy 
#	(BFGS, CG), code optimizations using LLVM interfaces, and options for unsupervised training of DBN's.
#
# usage : python neuralnet.py <traing_set> <testing_set> <file_type> <optimization_method> <maxiter> 
# 	<batch_size> <hidden_layer_sizes> <regularization_term> <learning_rate> 
# 	Usage Notes:
#	- The training and testing set are assumed to have the same number of features.  The algorithm will
#	automatically detect and handle multi-class classification problems.
#	- The file type can be either CSV or HDF, specified as "csv" and "hdf" respectively.
#	- Optimization method options are: "l-bfgs", "cg", None (Stochastic Gradient Descent)
#	- If the batch_size is set to -1, then batch optimization is used
#	- Hidden layer sizes must be separated by dashes i.e., "100-50-50"
#
# python_version  : 3.3.3
#==============================================================================

import sys
import math
import csv
import scipy.optimize as opti
import numpy as np
# np.seterr(all='raise')
from mlutils import mlutils as mlu

class NeuralNetClassifier():
	"""
	This class is responsible for all neural network classifier operations.  These operations include
	building a model from training data, class prediction for testing data, and printing the model.
	"""
	def __init__(self, method=None, maxiter=100, batch_size=-1, units=None, lmbda=0, 
				 alpha=100, beta=1000):
		"""
		Initialize neural network model based on user specifications.
		@parameters: 
			method - specifies the optimization method to use, "l-bfgs", "cg", or
						   None (defaults to SGD)
			maxiter - the maximum number of iterations allowed for training
			batch_size - the number of instance for each minibatch, -1 implies batch processing
			units - a sequence of integers separated by '.' such that each integer represents 
					 the number of units in a sequence of hidden layers.
			lmbda - the regularization term
			alpha - the numerator for the learning rate schedule (relevant for SGD only)
			beta - the denominator for the learning rate schedule (relevant for SGD only)
		"""
		# validate optimization method input
		if method not in ["l-bfgs", "cg", "sgd"]:
			mes = "Optimization Method Error: valid methods are 'l-bfgs', 'cg', and 'sgd'."
			raise Exception(mes)

		# parse hidden layer sizes
		self.units = [int(u) for u in units.split('-')] if units is not None else None

		# user defined parameters
		self.alpha = float(alpha) # numerator for learning rate calculation
		self.beta = float(beta) # denominator for learning rate calculation
		self.lmbda = float(lmbda) # regularization term
		self.maxiter = int(maxiter) # the maximum number of iterations through the data before stopping
		self.batch_size = int(batch_size) # for batch updates during gradient descent
		self.method = "SGD" if method is None else method # the method to use during optimization
		
		# internal optimization parameters
		self.gtol = 1e-7 # convergence measure
		self.init_epsilon = 1e-4 # for random initialization of theta values
		self.init_momentum = 0 #0.95 # dictates the weight of the previous update for momentum calculation
		self.momentum_decay = 0.9 # reduces the momentum effect with each iteration
		
		# internal classification parameters
		self.threshold = 0.5 # the class prediction threshold

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
		### Build Model Architecture ###
		if self.units is None:
			input_units = int(X.shape[1])
			self.units = [input_units, 2 * input_units]
		else:
			# insert the number of input units
			self.units.insert(0, int(X.shape[1]))

		# calculate and append the number of output units
		train_clss = np.unique(y) # get the unique elements of the labels array
		train_clss.sort()
		num_clss = len(train_clss)
		if num_clss == 2:
			self.units.append(1)
		else:
			self.units.append(num_clss)

		# format label to multiclass classification arrays
		y = mlu.multiclass_format(y, num_clss)

		# network structure parameters
		self.L = len(self.units) # the total number of layers including input and output layers

		# compute theta layer sizes and shapes
		self.sizes = [] # store theta layer divisions in the flattened theta array
		self.shapes = [] # store theta layer shapes for reshaping
		for i in range(len(self.units)-1):
			j_ = self.units[i+1]
			j = self.units[i]
			self.sizes.append(j_ * (j + 1))
			self.shapes.append((j_, j + 1))

		# randomly initialize weights for flattened theta array
		self.theta = np.random.rand(sum(self.sizes)) * (2 * self.init_epsilon) - self.init_epsilon

		### Perform Optimization ###
		self.n = X.shape[1] # the number of features
		self.m = X.shape[0] # the number of instances
		self.k = self.units[-1] # the number of output units
		self.momentum = self.init_momentum

		self.theta = self.minimize(self.method, self.cost, self.theta, X, y, self.jac, self.gtol)

	def minimize(self, method, cost, theta, X, y, jac, gtol):
		costs = [] # store cost for plotting
		mags = [] # store gradient magnitudes for plotting

		# check if batch processing is requested for advanced optimization techniques
		if self.batch_size == -1 and method == "l-bfgs":
			# L-BFGS-b optimization
			print("Performing batch optimization using L-BFGS-b.")
			theta, f, d = opti.fmin_l_bfgs_b(
					cost, theta, fprime=jac, args=(X, y), factr=10.0, 
					pgtol=1e-50, maxiter=self.maxiter, approx_grad=False, disp=1)
		elif self.batch_size == -1 and method == "cg":
			# conjugate gradient optimization
			print("Performing batch optimization using CG.")
			theta = opti.fmin_cg(
					cost, theta, fprime=jac, args=(X, y), 
					gtol=gtol, maxiter=self.maxiter, disp=1)
		else:
			# minibatch process and/or standard gradient descent was requested
			print("Performing minibatch optimization using", method, "with batch size", self.batch_size)

			# iterate through the data at most maxiter times, updating the theta for each feature also stop 
			# iterating if magnitude of the gradient is less than epsilon (convergence tolerance constant)
			for iteration in range(self.maxiter):
				# compute learning rate
				learning_rate = self.alpha / (self.beta + iteration)

				mags_tmp = [] # temporarily store magnitudes of each batch to calculate an average
				step = 0 # stores last update value for momentum calculations

				# iterate through batches
				for batch_count, (X_, y_) in enumerate(self.mini_batch(X, y)):
					if method == "l-bfgs":
						# L-BFGS-b optimization
						theta, f, d = opti.fmin_l_bfgs_b(
								cost, theta, fprime=jac, args=(X_, y_), 
								factr=10.0, pgtol=1e-50, maxiter=20, approx_grad=False)
					elif method == "cg":
						# Conjugate Gradient optimization
						theta = opti.fmin_cg(
								cost, theta, fprime=jac, args=(X_, y_), 
								gtol=1e-50, maxiter=3, disp=False)
					else: # Stochastic Gradient Descent
						# compute the cost
						costs.append(self.cost(theta, X_, y_))

						# compute the gradient for current batch
						D = self.jac(theta, X_, y_)

						# perform gradient checking (testing only)
						# print
						# grad_approx = self.estimate_gradient(X_, y_)
						# for i, ga in enumerate(grad_approx):
						# 	print i, D[i], ga

						# update theta parameters
						if iteration == 0 and batch_count == 0: # calculate first step to initiate momentum
							last_step = learning_rate * D
							self.theta -= last_step
						else: # use momentum
							correction = learning_rate * D
							step = (self.momentum * last_step) + ((1 - self.momentum) * correction)
							self.theta -= step
							last_step = step

						# calculate the magnitude of the gradient and check for convergence
						mag = np.linalg.norm(D)
						mags.append(mag)
						mags_tmp.append(mag)
						if gtol > mag:
							break
				
				# output iteration number and avg magnitude of the gradient if appropriate
				if len(mags_tmp) > 0:
					print("iteration", iteration, ":", np.mean(mags_tmp))
				else:
					print("iteration", iteration)

				# update momentum
				self.momentum = self.momentum_decay * self.momentum

		return theta #, costs, mags

	def mini_batch(self, X, y):
		"""
		Returns a generator object representing the X, y pairs for each minibatch.  Generates
		batches using the batch_size parameter.  A batch_size of -1 implies batch processing.
		"""
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

		# compute regularization
		for theta_j in thetas:
			theta_sum += (theta_j[:,1:] ** 2).sum()
		reg = (self.lmbda / (2 * m)) * theta_sum

		# compute the sum of the error
		cost_sum = 0
		for i, x in enumerate(X):
			a, h_x = self.forward_prop(x, thetas)
			for k in range(self.k):
				cost_sum += (np.dot(y[i][k], np.log(h_x[k])) + np.dot((1 - y[i][k]), np.log(1 - h_x[k]))).sum()

		# normalize and add regularization
		return ((- 1.0 / m) * cost_sum) + reg

	def jac(self, theta, X, y):
		m = X.shape[0] # number of instances

		# set the delta accumulator for gradient descent to 0
		self.delta = np.zeros((theta.shape))

		# get theta parameter arrays for each layer (dev note: used to be in for loop)
		thetas = self.unpack_parameters(theta)

		# iterate through instances and accumulate deltas
		for i, x in enumerate(X):
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
			x = a_ # populate x with new "features" for next iteration of activation calculations
			a = np.hstack((a, a_)) # record current layer activation values
		return a, a_

	def back_prop(self, x, y, a_, thetas):
		a = self.unpack_activations(a_) # activation values for each layer
		d = [a[-1] - y] # delta_L (prediction error in output layer)

		# iterate through layer activation values in reverse order computing delta_l
		for j in reversed(range(1, self.L - 1)):
			a_tmp = np.hstack((1, a[j-1]))
			d_tmp = (np.dot(thetas[j].T, d[0]) * (a_tmp * (1 - a_tmp)))[1:]
			d.insert(0, d_tmp)

		# use delta_l from each layer to back-propagate output layer error
		a.insert(0, x)
		delta = np.array(())
		for l in range(1, self.L):
			delta_l =  np.outer(d[l-1], np.hstack((1, a[l-1])))
			delta = np.hstack((delta, delta_l.flatten()))

		# add the gradient update of the current instance to the delta accumulator
		self.delta += delta

	def estimate_gradient(self, X, y):
		epsilon = .0001 # the the one-sided distance from the actual theta parameter value

		# compute the derivative estimate with respect to each theta parameter
		grad_approx = np.zeros(self.theta.shape)
		for i in range(len(self.theta)):
			# adjust the current theta parameter based on epsilon
			theta_plus = np.copy(self.theta)
			theta_plus[i] += epsilon
			theta_minus = np.copy(self.theta)
			theta_minus[i] -= epsilon

			# compute the two-sided difference
			cost_plus = self.cost(theta_plus, X, y)
			cost_minus = self.cost(theta_minus, X, y)
			grad_approx[i] = (cost_plus - cost_minus) / (2 * epsilon)

		return grad_approx

	def compute_activation(self, z):
		return np.divide(1.0 , (1 + np.exp(- z)))

	# =====================================================================================
	# Model Architecture Utilities

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

	def get_cost(self, X, y):
		"""
		Returns the output of the cost function for the data instances and labels (X, y)
		given the model's current parameter values.
		"""
		return self.cost(self.theta, X, y)

	# ==================================================================================
	# Model Output

	# TODO
	# def print_model(self, features, model_file):
	# 	# wite the parameter values corresponding to each feature to the given model file
	# 	with open(model_file, 'w') as mf:
	# 		for i in range(self.n):
	# 			if i == 0:
	# 				mf.write('%f\n' % (self.theta[i]))
	# 			else:
	# 				mf.write('%s %f\n' % (features[i-1], self.theta[i]))


def main(train_file, test_file, load_method="csv", opti_method=None, maxiter=100, 
		 batch_size=-1, units=None, lmbda=0, alpha=100, beta=1000):
	"""
	Manages files and operations for the neural network model creation, training, and testing.
	@parameters: 
		load_method - the dataset file format, either "csv" or "hdf"
		opti_method - specifies the optimization method to use, "l-bfgs", "cg", or
					   None (defaults to SGD)
		maxiter - the maximum number of iterations allowed for training
		batch_size - the number of instance for each minibatch, -1 implies batch processing
		units - a sequence of integers separated by '.' such that each integer represents 
				 the number of units in a sequence of hidden layers.
		lmbda - the regularization term
		alpha - the numerator for the learning rate schedule (relevant for SGD only)
		beta - the denominator for the learning rate schedule (relevant for SGD only)
	"""
	# open and load csv files
	if load_method == "csv":
		X_train, y_train = mlu.load_csv(train_file, True) # load and shuffle training set
		X_test, y_test = mlu.load_csv(test_file)
	elif load_method == "hdf":
		X_train, y_train = mlu.loadh(train_file, True) # load and shuffle training set
		X_test, y_test = mlu.loadh(test_file)
	else:
		raise Exception("Dataset file type not recognized: acceptable formats are 'csv' and 'hfd'. \
						 Specify method with second argument.")

	# perform feature scaling
	X_train = mlu.scale_features(X_train, 0.0, 1.0)
	X_test = mlu.scale_features(X_test, 0.0, 1.0)

	# create the neural network classifier using the training data
	NNC = NeuralNetClassifier(opti_method, maxiter, batch_size, units, lmbda, alpha, beta)
	print("\nCreated a neural network classifier =", NNC)

	# fit the model to the loaded training data
	print("\nFitting the training data...")
	# costs, mags = NNC.fit(X_train, y_train)
	NNC.fit(X_train, y_train)

	# predict the results for the test data
	print("\nGenerating probability prediction for the test data...")
	y_pred = NNC.predict(X_test)

	### print the classification results ###
	print("\nThe probabilities for each instance in the test set are:\n")
	for prob in NNC.predict_proba(X_test):
		print(prob)
	# print simple precision metric to the console
	print('Accuracy:  ' + str(mlu.compute_accuracy(y_test, y_pred)))

if __name__ == '__main__':
	"""
	The main function is called when neuralnet.py is run from the command line with arguments.
	"""
	args = sys.argv[1:] # get arguments from the command line
	main( *args )
