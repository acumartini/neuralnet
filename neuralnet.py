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

		# calculate theta layer indices and shapes
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

	def fit(self, X_, y):
		"""
		This function optimizes the parameters for the neural network classification model from 
		training data
		@post: parameter(theta) optimized by forward and back propagation
		"""
		X = self.add_ones(X_) # prepend ones to training set for theta_0 calculations
		self.n = X.shape[1] # the number of features
		self.m = X.shape[0] # the number of instances

		# iterate through the data at most max_iter times, updating the theta for each feature
		# also stop iterating if error is less than epsilon (convergence tolerance constant)
		print "iter | magnitude of the gradient"
		for iteration in xrange(self.max_iter):
			for i in xrange(self.m):
				# calculate the activation values
				a = self.forward_prop(X[i])

				# back propagate the error
				delta = self.back_prop(a)
			
				# calculate Delta accumulations
				self.D += np.dot(delta, a)

			# compute the partial derivative terms with regularization
			# grad = ...

			# perform gradient checking
			grad_estimate = self.estimate_gradient()

			# update theta parameters
			# self.theta = ...

			# calculate the magnitude of the gradient and check for convergence
			grad_mag = np.linalg.norm(grad)
			if self.epsilon > grad_mag:
				break
			
			print iteration, ":", grad_mag

	def forward_prop(self, x):
		a = None
		j = 1 # store current processing layer
		i = 0 # store flattened theta array index value from previous calculation
		for I,s in zip(self.indices, self.shapes):
			print i, i + I, s[0], s[1]
			theta_j = self.theta[i:i + I]
			theta_j = theta_j.reshape(s[0], s[1])
			i = I
			print theta_j, theta_j.shape

			# calc z value
			# print x
			z_j = np.dot(theta_j, x)
			print z_j
			a = self.get_activation(z_j)
			print "a:", a
			x = np.array([1])
			x = np.hstack((x, a))
			# print x
			# raise

			# raise

	def back_prop(self, a):
		pass

	def estimate_gradient(self):
		pass

	def get_activation(self, z):
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


def load_csv(data, features=False):
	"""
	Loads the csv files into numpy arrays.
	@parameters: data The data file in csv format to be loaded
				 features True => the data file includes features as the first row 
	@returns: feature_names - None if features == False, else a list of feature names
			  X - numpy array of data instances with dtype=float
			  y - numpy array of labels
	"""
	print "Loading data from", data

	# first create attribute names tuple and labels array
	if features:
		with open(data, 'rb') as csv_file:
			reader = csv.reader(csv_file)
			feature_names = list(reader.next())
			feature_names.pop() # remove the 'class' feature name
	else:
		feature_names = None

	# load the csv data into a numpy array
	X = np.loadtxt(data, delimiter=",", dtype='float', skiprows=1)
	y = X[:,-1:] # get only the labels
	y = y.flatten() # make the single column 1 dimensional
	y = y.astype(int)
	X = X[:,:-1] # remove the labels column from the data array
	return feature_names, X, y

def scale_features(X):
	# scales all features in dataset X to values between 0.0 and 1.0
	new_min, new_max = 0.0, 1.0
	A_ = X.T
	for i, column in enumerate(A_):
		old_max = column.max()
		old_min = column.min()
		A_[i] = ((column - old_min) / (old_max - old_min + 0.000001) * (new_max - new_min)) + new_min
	X = A_.T

	# vectorized implementation slightly off, still debugging...
	# X_min, X_max = X.min(0), X.max(0)
	# new_min, new_max = 0.0, 1.0
	# X = (((X - X_min) / (X_max - X_min)) * (new_max - new_min)) + new_min

def get_accuracy(y_test, y_pred):
	"""
	@returns: The precision of the classifier, (correct labels / instance count)
	"""
	correct = 0
	for i, pred in enumerate(y_pred):
		if int(pred) == y_test[i]:
			correct += 1
	return float(correct) / y_test.size


def main(train_file, test_file, layers=1, units=None, lmbda=0, max_iter=10000):
	"""
	Manages files and operations for logistic regression model creation
	"""
	# open and load csv files
	features, X_train, y_train = load_csv(train_file, True)
	dummy, X_test, y_test = load_csv(test_file, True)

	# scale features to encourage gradient descent convergence
	scale_features(X_train)
	scale_features(X_test)

	# get units list
	input_units = int(X_train.shape[1])
	units_ = [input_units]
	if units is None:
		units_.extend([2 * input_units] * int(layers))
	else:
		units_.extend([int(u) for u in units.split('.')])

	# binary classification for now
	units_.append(1)

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