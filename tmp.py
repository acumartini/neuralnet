# Adam Martini
# Start Date: 1-7-14

import sys
import csv
import numpy as np
# np.seterr(all='raise')

class NeuralNetClassifier():
	"""
	This class is responsible for all neural network classifier operations.  These operations include
	building a model from training data, class prediction for testing data, and printing the model.
	"""
	def __init__(self, units, lmbda, alpha, max_iter):
		self.units = units # the number of units in each hidden layer
		self.L = len(units) # the total number of layers including input and output layers
		self.alpha = float(alpha) # learning rate for gradient descent
		self.lmbda = float(lmbda) # regularization term
		self.max_iter = int(max_iter) # the maximum number of iterations through the data before stopping
		self.epsilon = 0.00001 # convergence measure
		self.init_epsilon = 0.0001 # for random initialization of theta values
		self.threshold = 0.5 # the class prediction threshold

		# build network architecture by computing theta layer indices and shapes
		self.indices = [] # store theta layer divisions in the flattened theta array
		self.shapes = [] # store theta layer shapes for reshaping
		for i in xrange(len(self.units)-1):
			j_ = self.units[i+1]
			j = self.units[i]
			self.indices.append(j_ * (j + 1))
			self.shapes.append((j_, j + 1))

		# randomly initialize weights for flattened theta array
		self.theta = np.random.rand(sum(self.indices)) * (2 * self.init_epsilon) - self.init_epsilon

	def __str__(self):
		return "<Neural Network Classifier Instance: units=" + str(self.units) + ">\n"

	def fit(self, X, y):
		"""
		This function optimizes the parameters for the neural network classification model from 
		training data
		@post: parameter(theta) optimized by forward and back propagation
		"""
		self.n = X.shape[1] # the number of features
		self.m = X.shape[0] # the number of instances
		self.k = self.units[-1] # the numer of output units

		# iterate through the data at most max_iter times, updating the theta for each feature
		# also stop iterating if error is less than epsilon (convergence tolerance constant)
		print "iter | magnitude of the gradient"
		for iteration in xrange(self.max_iter):
			self.delta = np.zeros((self.theta.shape)) # the delta accumulator for gradient descent
			for i in xrange(self.m):
				# get theta parameter arrays for each layer
				thetas = self.get_parameter_arrays(self.theta)

				# calculate the activation values
				a, h_x = self.forward_prop(X[i], thetas)

				# back propagate the error
				self.back_prop(X[i], y[i], a, thetas)
			
			# compute the partial derivative terms with regularization
			D = (1.0/self.m * self.delta) + (self.lmbda * self.theta)
			# D = 1.0/self.m * (self.delta + (self.lmbda * self.theta))
			# print D
			# raise

			# perform gradient checking
			# grad_estimate = self.estimate_gradient(X, y)
			# print grad_estimate, np.linalg.norm(self.D)

			# update theta parameters
			self.theta -= self.alpha * D
			# print self.theta

			# calculate the magnitude of the gradient and check for convergence
			mag = np.linalg.norm(D)
			if self.epsilon > mag:
				break
			
			print iteration, ":", mag

	def forward_prop(self, x_, thetas=None):
		a = np.array(()) # store activation values for each hidden and output layer unit

		# iterate through each layer in the network computing and forward propagating activation values
		x = x_ # preserve original x
		if thetas is None:
			thetas = self.get_parameter_arrays(self.theta)
		for theta_j in thetas:
			x = np.hstack((1, x)) # add bias unit with value 1.0
			a_ = self.compute_activation(np.dot(theta_j, x)) # the current layer's activation values
			x = a_ # populate x with new "features" for next iteration of activation calcs
			a = np.hstack((a, a_)) # record current layer activation values
		return a, a_

	def get_parameter_arrays(self, param):
		params = []
		i = 0 # store flattened theta array index value from previous iteration
		for j,s in zip(self.indices, self.shapes):
			params.append(param[i:i + j].reshape(s[0], s[1])) # get current layers theta matrix
			i = j # record the flattened array index for the end of current layer
		return params

	def get_activation_arrays(self, a_):
		a = []
		i = 0 # store flattened activation array index value from previous iteration
		for j in self.units[1:]:
			a.append(a_[i:i+j]) # append current activation layer values
			# a.append(a_[i:i+j][:, np.newaxis]) # append current activation layer values
			i = j
		return a

	def back_prop(self, x, y, a_, thetas):
		a = self.get_activation_arrays(a_)

		# print "thetas"
		# for t in thetas:
		# 	print t	
		# print "activations"
		# for a__ in a:
		# 	print a__
		# print "y", y

		d = [a[-1] - y] # delta_L
		# iterate through layer activation values in reverse order computing d
		for j in reversed(xrange(1, self.L - 1)):
			# print "j", j
			# print "a[j]", a[j]
			# print "thetas[j].T", thetas[j].T
			# print "d[0]", d[0]
			# print "(a[j] * (1 - a[j]))", (a[j] * (1 - a[j]))
			# print np.dot(thetas[j].T, d[0])

			if j == self.L - 2:
				a_tmp = a[j]
			else:
				a_tmp = np.hstack((1, a[j]))
			# print np.dot(thetas[j].T, d[0])
			# print (a_tmp * (1 - a_tmp))
			d_tmp = (np.dot(thetas[j].T, d[0]) * (a_tmp * (1 - a_tmp)))[1:]
			d.insert(0, d_tmp)
			
			# d.insert(0, np.dot(thetas[j].T, d[0]) * (a[j] * (1 - a[j])))
			# if j == self.L - 2:
			# 	# print "d[0]", d[0]
			# 	# print np.dot(thetas[j].T, d[0])
			# 	# print "(a[j] * (1 - a[j])", (a[j] * (1 - a[j]))
			# 	# print "delta_j", np.dot(thetas[j].T, d[0]) * (a[j] * (1 - a[j]))
				# d_tmp = (np.dot(thetas[j].T, d[0]) * (a[j] * (1 - a[j])))[1:]
				# d.insert(0, d_tmp)
			# else:
			# 	# print "d[0][1:]", d[0][1:]
			# 	# print np.dot(thetas[j].T, d[0][1:])
			# 	# print "(a[j] * (1 - a[j])", np.hstack((1, (a[j] * (1 - a[j]))))
			# 	# print "delta_j", np.dot(thetas[j].T, d[0][1:]) * np.hstack((1, (a[j] * (1 - a[j]))))
				# a_tmp = np.hstack((1, a[j]))
				# d.insert(0, np.dot(thetas[j].T, d[0][1:]) * (a_tmp * (1 - a_tmp)))
			# deltas.insert(0, np.dot(thetas[j].T, deltas[0]) * np.hstack((1, (a[j] * (1 - a[j])))))
		
		# print "d"
		# for d__ in d:
		# 	print d__
		# raise

		# for i, params in enumerate(reversed(zip(activations[:-1], thetas[1:]))):
		# 	a, theta = params
		# 	if i == 0:
		# 		d_tmp = deltas[i]
		# 	else:
		# 		d_tmp = deltas[i][1:]
				
		# 	print "theta", theta
		# 	print "deltas[i]", d_tmp
		# 	# print np.dot(theta.T, deltas[i])
		# 	print np.dot(theta.T, d_tmp)
		# 	print "a", a
		# 	print np.hstack((1, (a * (1 - a))))
		# 	# print np.dot(theta.T, deltas[i]) * np.hstack((1, (a * (1 - a))))
		# 	print np.dot(theta.T, d_tmp) * np.hstack((1, (a * (1 - a))))
		# 	print
		# 	deltas.append(np.dot(theta.T, deltas[i]) * np.hstack((1, (a * (1 - a))))) # delta_j
		# 	deltas.append(np.dot(theta.T, d_tmp) * np.hstack((1, (a * (1 - a))))) # delta_j

		# print
		# deltas.reverse()
		# print "thetas", thetas

		a.insert(0, x)
		# print "activations:", activations

		delta = np.array(())
		for l in xrange(1, self.L):
			# print
			# print "l", l
			# print thetas[l-1]
			# print "d[l-1]", d[l-1]
			# print "d[l-1][1:]", d[l-1][1:]
			# print "a[l-1]", a[l-1]
			# print np.outer(d[l-1], np.hstack((1, a[l-1])))
			delta_l =  np.outer(d[l-1], np.hstack((1, a[l-1])))
			# print delta_l

			# delta_l = np.dot(d[l-1], np.hstack((1, a[l-1]))[:, np.newaxis])
			# if l == self.L - 1:
				# print "np.outer(d[l-1], np.hstack((1, a[l-1]))", np.outer(d[l-1], np.hstack((1, a[l-1])))
				# delta_l =  np.outer(d[l-1], np.hstack((1, a[l-1])))
			# else:
				# print "np.outer(d[l-1][1:], np.hstack((1, a[l-1]))", np.outer(d[l-1][1:], np.hstack((1, a[l-1])))
				# delta_l =  np.outer(d[l-1][1:], np.hstack((1, a[l-1])))

			delta = np.hstack((delta, delta_l.flatten()))

		# raise
		# layer = 1
		# for i, params in enumerate(zip(deltas, a)):
		# 	d, a = params
		# 	if layer != len(activations) - 1:
		# 		d_tmp = d[1:, np.newaxis] * np.hstack((1,a))
		# 	else:
		# 		d_tmp = d[:, np.newaxis] * np.hstack((1,a))
		# 	delta = np.hstack((delta, d_tmp.flatten()))
		# 	layer += 1

		# add backprob values to the delta accumulator
		# print delta
		# raise
		self.delta += delta

	def compute_cost(self, theta, X, y):
		# compute the cost function J(theta) using the regularization term lmbda
		theta_sum = 0
		for theta_j in self.get_parameter_arrays(theta):
			theta_sum += (theta_j[:,1:] ** 2).sum()
		reg = (self.lmbda / (2 * self.m)) * theta_sum

		# print y, y.shape
		# print np.log(h_x), np.log(h_x).shape

		m_sum = 0
		for i, x in enumerate(X):
			a, h_x = self.forward_prop(x)
			m_sum += (np.dot(y[i], np.log(h_x)) + np.dot((1 - y[i]), np.log(1 - h_x))).sum()
		return 1.0/self.m * m_sum + reg
		# return 1.0/self.m * (np.dot(y[:, np.newaxis], np.log(h_x)) + np.dot((1 - y)[:, np.newaxis], np.log(1 - h_x))).sum() + reg

	def estimate_gradient(self, X, y):
		# def estimation_helper(theta, x, y):
		# 	a, h_x = self.forward_prop(x)
		# 	return self.compute_cost(theta, y, h_x)

		epsilon = 0.0001

		# calculate the cost of theta +/- epsilson
		plus_cost = self.compute_cost(self.theta + epsilon, X, y)
		minus_cost = self.compute_cost(self.theta - epsilon, X , y)
		print plus_cost, minus_cost

		# compute the slope estimate of the gradient
		return (plus_cost - minus_cost) / (2 * epsilon)

	def compute_activation(self, z):
		# for i, v in enumerate(z):
		# 	if v < -700:
		# 		z[i] = -700
		return 1.0 / (1 + np.exp(- z))
		# for i, v in enumerate(a):
		# 	if float(v) == float('inf'):
		# 		a[i] = 1.0
		# return a

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

	def add_ones(self, X):
		# prepend a column of 1's to dataset X to enable theta_0 calculations
		return np.hstack((np.zeros(shape=(X.shape[0],1), dtype='float') + 1, X))

	# def print_model(self, features, model_file):
	# 	# wite the parameter values corresponding to each feature to the given model file
	# 	with open(model_file, 'w') as mf:
	# 		for i in xrange(self.n):
	# 			if i == 0:
	# 				mf.write('%f\n' % (self.theta[i]))
	# 			else:
	# 				mf.write('%s %f\n' % (features[i-1], self.theta[i]))


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
		y_[i][lable] = 1.0
	return y_

def get_accuracy(y_test, y_pred):
	"""
	@returns: The precision of the classifier, (correct labels / instance count)
	"""
	correct = 0
	for i, pred in enumerate(y_pred):
		if int(pred) == y_test[i]:
			correct += 1
	return float(correct) / y_test.size


def main(train_file, test_file, alpha=0.01, max_iter=10000, lmbda=0, units=None):
	"""
	Manages files and operations for the neural network model creation, training, and testing.
	@parameters: alpha - the learning rate for gradient descent
				 max_iter - the maximum number of iterations allowed for training
				 lmbda - the regularization term
				 units - a sequence of integers separated by '.' sunch that each integer
				 represents the numer of units in a sequence of hidden layers.
	"""
	# open and load csv files
	X_train, y_train = load_csv(train_file)
	X_test, y_test = load_csv(test_file)

	# scale features to encourage gradient descent convergence
	X_train = scale_features(X_train, 0.0, 1.0)
	X_test = scale_features(X_test, 0.0, 1.0)

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
		print "Warning: Training and testing set labels do not agree."
	
	# record the number of output units
	num_clss = len(train_clss)
	if num_clss == 2:
		units_.append(1)
	else:
		units_.append(num_clss) # record the number of output units

		# format dataset labels to multiclass classification arrays
		y_train = multiclass_format(y_train, num_clss)
		y_test_ = multiclass_format(y_test, num_clss)

	# create the neural network classifier using the training data
	NNC = NeuralNetClassifier(units_, lmbda, alpha, max_iter)
	print "\nCreated a neural network classifier =", NNC

	# fit the model to the loaded training data
	print "Fitting the training data...\n"
	NNC.fit(X_train, y_train)

	# predict the results for the test data
	print "Generating probability prediction for the test data...\n"
	y_pred = NNC.predict(X_test)

	### print the classification results ###
	print "The probabilities for each instance in the test set are:\n"
	for prob in NNC.predict_proba(X_test):
		print prob
	# print simple precision metric to the console
	print('Accuracy:  ' + str(get_accuracy(y_test, y_pred)))
	
	# write the model to the model file
	# NNC.print_model(features, model_file)


if __name__ == '__main__':
	"""
	The main function is called when neuralnet.py is run from the command line with arguments.
	"""
	args = sys.argv[1:] # get arguments from the command line
	main( *args )