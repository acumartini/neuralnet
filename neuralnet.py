# Adam Martini
# Start Date: 1-7-14

import sys
import csv
import numpy as np

class NeuralNetClassifier():
	"""
	This class is responsible for all neural network classifier operations.  These operations include
	building a model from training data, class prediction for testing data, and printing the model.
	"""
	def __init__(self, layers, units, lmbda, max_iter):
		self.layers = int(layers) # the number of hidden layers in the neural network
		self.units = units # the number of units in each hidden layer
		self.lmbda = float(lmbda) # regularization term
		self.max_iter = int(max_iter) # the maximum number of iterations through the data before stopping
		self.epsilon = 0.00001 # convergence measure
		self.threshold = 0.5 # the class prediction threshold

		# build network architecture by computing theta layer indices and shapes
		self.indices = [] # store theta layer divisions in the flattened theta array
		self.shapes = [] # store theta layer shapes for reshaping
		for i in xrange(len(self.units)-1):
			j_ = self.units[i+1]
			j = self.units[i]
			self.indices.append(j_ * (j + 1))
			self.shapes.append((j_, j + 1))
		print self.indices, self.shapes

		# randomly initialize weights for flattened theta array
		self.theta = np.random.rand(sum(self.indices)) - 0.5
		print self.theta, self.theta.shape

	def __str__(self):
		return "<Neural Network Classifier Instance: layers=" + str(self.layers) + \
			   ", units=" + str(self.units) + ">\n"

	def fit(self, X, y):
		"""
		This function optimizes the parameters for the neural network classification model from 
		training data
		@post: parameter(theta) optimized by forward and back propagation
		"""
		self.n = X.shape[1] # the number of features
		self.m = X.shape[0] # the number of instances
		self.k = self.units[-1] # the numer of output units
		self.D = np.zeros((self.theta.shape)) # the delta accumulator for gradient descent

		# iterate through the data at most max_iter times, updating the theta for each feature
		# also stop iterating if error is less than epsilon (convergence tolerance constant)
		print "iter | magnitude of the gradient"
		for iteration in xrange(self.max_iter):
			for i in xrange(self.m):
				# calculate the activation values
				a = self.forward_prop(X[i])

				# calculate the cost
				h_x = a[:-(self.k + 1)] if self.k > 1 else a[-1] # the activation values on the output units
				cost = self.compute_cost(y, h_x)

				# back propagate the error
				delta = self.back_prop(X[i], y[i], a)
			
			# compute the partial derivative terms with regularization
			self.D = (1.0/self.m * delta) + (self.lmbda * self.theta)
			print self.D

			# perform gradient checking
			# grad_estimate = self.estimate_gradient()

			# update theta parameters
			# self.theta = ...

			# calculate the magnitude of the gradient and check for convergence
			grad_mag = np.linalg.norm(grad)
			if self.epsilon > grad_mag:
				break
			
			print iteration, ":", grad_mag

	def forward_prop(self, x_):
		a = np.array(()) # store activation values for each hidden and output layer unit

		# iterate through each layer in the network computing and forward propagating activation values
		x = x_ # preserve original x
		for theta_j in self.get_parameter_arrays(self.theta):
			x = np.hstack((1, x)) # add bias unit value
			a_ = self.compute_activation(np.dot(theta_j, x)) # the current layer's activation values
			x = a_ # populate x with new "features" for next iteration of activation calcs
			a = np.hstack((a, a_)) # record current layer activation values
			print "a_:", a_
			print "a:", a
		return a

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
			i = j
		return a

	def back_prop(self, x, y, a_):
		deltas = []
		activations = self.get_activation_arrays(a_)
		thetas = self.get_parameter_arrays(self.theta)
		
		deltas = [activations[-1] - y] # delta_L
		# iterate through layer activation values in reverse order to calulate delta_j
		for i, params in enumerate(reversed(zip(activations[:-1], thetas[1:]))):
			a, theta = params
			print np.dot(theta.T, deltas[i]) * np.hstack((1, (a * (1 - a))))
			deltas.append(np.dot(theta.T, deltas[i]) * np.hstack((1, (a * (1 - a))))) # delta_j

		# debugging backprop...
		print
		deltas.reverse()
		big_deltas = self.get_parameter_arrays(self.D)
		print "Deltas", big_deltas
		print "thetas", thetas
		print "deltas", deltas
		activations.insert(0, x)
		print "activations:", activations

		D_ = np.array(())
		for D, d, a in zip(big_deltas, deltas, activations):
			print D
			print np.hstack((1,a))[:, np.newaxis]
			print d[1:]
			D_tmp = D + d[1:, np.newaxis] * np.hstack((1,a))
			print D_tmp
			D_ = np.hstack((D_, D_tmp.flatten()))
		print D_
		return D_

	def compute_cost(self, y, h_x):
		# compute the cost function J(theta) using the regularization term lmbda
		theta_sum = 0
		for theta_j in self.get_parameter_arrays(self.theta):
			theta_sum += (theta_j[:,1:] ** 2).sum()
		reg = (self.lmbda / (2 * self.m)) * theta_sum

		return 1.0/self.m * (np.dot(y, np.log(h_x)) + np.dot(1 - y, np.log(1 - h_x))).sum() + reg

	def estimate_gradient(self):
		pass

	def compute_activation(self, z):
		return 1.0 / (1 + np.exp(- z))

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
		X_ = self.add_ones(X)
		return self.get_proba(X_)

	def predict(self, X):
		"""
		Classifies a set of data instances X based on the set of trained feature theta.
		@parameters: X - array-like of shape = [n_samples, n_features]
		    		 The input samples.
		@returns: y_pred - list of shape = [n_samples]
				  The predicted class label for each instance.
		"""
		y_pred = [proba > self.threshold for proba in self.predict_proba(X)]
		return np.array(y_pred)

	def add_ones(self, X):
		# prepend a column of 1's to dataset X to enable theta_0 calculations
		return np.hstack((np.zeros(shape=(X.shape[0],1), dtype='float') + 1, X))

	def print_model(self, features, model_file):
		# wite the parameter values corresponding to each feature to the given model file
		with open(model_file, 'w') as mf:
			for i in xrange(self.n):
				if i == 0:
					mf.write('%f\n' % (self.theta[i]))
				else:
					mf.write('%s %f\n' % (features[i-1], self.theta[i]))


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
	y = y.flatten() # make the single column 1 dimensional
	# y = y.astype(int)
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


def main(train_file, test_file, lmbda=0, layers=1, units=None, max_iter=10000):
	"""
	Manages files and operations for logistic regression model creation
	"""
	# open and load csv files
	X_train, y_train = load_csv(train_file)
	X_test, y_test = load_csv(test_file)

	# scale features to encourage gradient descent convergence
	scale_features(X_train, 0.0, 1.0)
	scale_features(X_test, 0.0, 1.0)

	# get units list
	input_units = int(X_train.shape[1])
	units_ = [input_units]
	if units is None:
		units_.extend([2 * input_units] * int(layers))
	else:
		units_.extend([int(u) for u in units.split('.')])

	# calculate the number of output units
	print y_train, y_test
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
		y_test = multiclass_format(y_test, num_clss)

	print layers, units_
	# create the neural network classifier using the training data
	NNC = NeuralNetClassifier(layers, units_, lmbda, max_iter)
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
	NNC.print_model(features, model_file)


if __name__ == '__main__':
	"""
	The main function is called when neuralnet.py is run from the command line with arguments.
	"""
	args = sys.argv[1:] # get arguments from the command line
	main( *args )