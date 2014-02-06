# Adam Martini
# ml_suite
# 2-4-14

import numpy as np
import csv
import sys
import math
from time import time

import mlutils as mlu

class InfoRank():
	"""
	Performs information gain ranking of labeled binary data.
	"""
	def __init__(self, criterion='entropy', chi_squared=0.0):
		# initialize internal parameters
		self.chi_squared_min = chi_squared
		self.criterion = criterion

	def get_info_rank(self, top, X, y):
		"""
		@returns: rank_list - a sorted list of features/original_index pairs.
		"""
		self.top = top
		self.feat_indices = [x for x in xrange(X.shape[1])]
		self.feat_cards = [] # stores value cardinalities for each feature

		# calculate domain cardinailties for each feature and save index in the name hash
		for i, col in enumerate(X.T):
			num_val = 0
			vals = set(col) # dev note: was []
			# for val in col:
			# 	if val not in vals:
			# 		num_val += 1
			# 		vals.append(val)
			self.feat_cards.append(len(vals))
		
		# create rank list, zip features, and sort
		rank_list = self.choose_best_features(X, y)
		rank_list = zip(gains, self.feat_indices)
		rank_list.sort()
		rank_list.reverse()

		# return the 'top' ranked features
		if self.top == 'ALL':
			return rank_list
		return rank_list[:self.top]

	def choose_best_feature(self, X, y):
		"""
		@returns: The feature with the highest information gain among the remaining
				  choices in the data array
	  	"""
	  	# check for final column
	  	if X.shape[1] == 1:
	  		return None
  		if self.criterion == 'entropy':
  			# get the overall info
  			info_y = self.info(float(np.sum(y==1)), float(np.sum(y==0)))

  			# get a list of values of the information gain of each potential split
		  	gains = []
		  	for i, feat in enumerate(self.feat_cards):
		  		# get entropy, filtering gains based chi-squared test
		  		feat_card = xrange(self.feat_cards[i])
	  			split_entropy, chi_squared = self.get_entropy(feat_card, X.transpose()[i], y)
	  			gain = info_y - split_entropy if chi_squared else 0.0
	  			gains.append(gain)

			# return the index of the maximum information gain value
			return gains

  		elif self.criterion == 'gini':
			# get the overall impurity and probabilities
			p = float(np.sum(y==1))
			n = float(np.sum(y==0))
			gini_y = self.get_gini((n, p))
			prob_0 = n / y.size
			prob_1 = p / y.size

			# get a list of values of the average impurity for each potential split
			ginis = []
		  	for i, feat in enumerate(self.feat_cards):
		  		# get entropy, filtering gains based chi-squared test
		  		feat_card = xrange(self.feat_cards[i])
	  			split_gini, chi_squared = self.get_split_gini(feat_card, X.transpose()[i], y, prob_0, prob_1)
	  			gini = gini_y - split_gini if chi_squared else 0.0
	  			ginis.append(gini)

			# return the maximum gini_index
	  		return ginis

	  	# TODO: add variable-domain size capability for KL
	  	elif self.criterion == 'KL':
			# get the overall info and probabilities 
			prob_0 = float(np.sum(y==0)) / y.size
			prob_1 = float(np.sum(y==1)) / y.size
			# get a list of values of the average weights
			kl_divs = []
			for i, feat in enumerate(self.feat_cards):
		  		# get entropy, filtering gains based chi-squared test
		  		feat_card = xrange(self.feat_cards[i])
	  			kl_div, chi_squared = self.get_kl_div(feat_card, X.transpose()[i], y, prob_0, prob_1)
	  			kl_div = kl_div if chi_squared else 0.0
	  			kl_divs.append(kl_div)
			
			# noramlize and return
			kl_divs = self.normalize(kl_divs)
			return kl_divs

	def normalize(self, divs):
		n_const = sum(divs) / len(self.features)
		return [weight/n_const for weight in divs]

	def klm_helper(self, prob_0, prob_1):
		if prob_0 == 0.0 and prob_1 == 0.0:
			return 0.0
		if prob_0 == 0.0:
			return (prob_1 * math.log(prob_1, 2))
		if prob_1 == 0.0:
			return (prob_0 * math.log(prob_0, 2))
		return (prob_0 * math.log(prob_0, 2)) + (prob_1 * math.log(prob_1, 2))

	def get_kl_div(self, A, y, prob_C0, prob_C1):
		prob_0, p0, n0, prob_1, p1, n1, y0, y1 = self.get_split_probs(A, y)
		# get the kullback-leibler measures for a=1 and a=0
		klm_0 = self.get_klm(prob_C0, prob_C1, p0, n0)
		klm_1 = self.get_klm(prob_C0, prob_C1, p1, n1)

		# get the chi-squared test result
		n = float(np.sum(y==0))
		p = float(np.sum(y==1))
		chi_squared_r = self.chi_squared_test(p, n, prob_0, prob_1, p0, n0, p1, n1)

		# get the split info
		split_info = self.info(float(y1.size), float(y0.size))

		# return (weight, chi_squared_r)
		# return ((prob_0 * klm_0 + prob_1 * klm_1) / split_info, chi_squared_r)
		return ((prob_0 * klm_0 + prob_1 * klm_1), chi_squared_r)

	def get_klm(self, prob_C0, prob_C1, p, n):
		if prob_C0 == 0.0 and prob_C1 == 0.0:
			return 0.0
		prob_p = float(p) / (p + n)
		prob_n = float(n) / (p + n)
		if prob_p == 0.0 and prob_n == 0.0:
			return 0.0
		if prob_p == 0.0:
			return prob_C0 * math.log(prob_n/prob_C0)
		if prob_n == 0.0:
			return prob_C1 * math.log(prob_p/prob_C1)
		return prob_p * math.log(prob_p/prob_C1) + prob_n * math.log(prob_n/prob_C0)

	def get_gini(self, split_count):
		"""
		Calculates gini index for set of labels y.  At the self level to reduce repetitions
		"""
		p, n = split_count
		if p + n == 0:
			return 0.0

		# calc probabilities
		prob_0 = n / (p + n)
		prob_1 = p / (p + n)

		# return the gini calc
		return 1 - (prob_0**2 + prob_1**2)

	def get_split_gini(self, card, A, y, prob_0, prob_1):
		"""
		Splits at the attribute A and return the average impurity of the split.
		@reutrn: gini_A, the average of the gini values resulting from a split at A
		"""
		# get the impurity of the splitting attribute A
		probas, prob_counts = self.get_split_probs(card, A, y)
		
		# get the average resulting impurity of spliting at attribute A
		gini = 0
		for val in card:
			gini += probas[val] * self.get_gini(prob_counts[val])

		# get the chi-squared test result
		n = float(np.sum(y==0))
		p = float(np.sum(y==1))
		chi_squared = self.chi_squared_test(p, n, card, probas, prob_counts)
		
		# return the spit gini value with the chi-squared test result
		return gini, chi_squared

	def get_split_probs(self, card, B, y):
		"""
		This function is placed at the "self" level because it is used by both
		split criterion, entropy and gini
		@returns: The probabilities for geting a 0 or 1 from a split attribute B
				  and the counts of 0/1 labels for each split
	  	"""
		# split attribute B into sub-lists conditioned upon the possible values of B
		y_splits = []
		count_hash = {}
		for val in card:
			y_splits.append(np.zeros(shape=np.sum(B==val), dtype='int'))
			count_hash[val] = 0
		for index, val in enumerate(B):
			y_splits[val][count_hash[val]] = y[index]
			count_hash[val] += 1

		# get the resulting p/n counts in A given b=val for val in {b_vals}
		prob_counts = []
		for var in card:
			split_count = []
			for target in [0,1]:
				split_count.append(float(np.sum(y_splits[var]==target)))
			prob_counts.append(copy.deepcopy(split_count))

		# get the resulting probabilities for P(A | val) for val in {b_vals}
		probas = []
		for val in card:
			probas.append(float(y_splits[val].shape[0])/(y.size)) # the probability of A given b=val

		return probas, prob_counts

	def chi_squared_test(self, p, n, card, probas, prob_counts):
		"""
		This helper function for chi_squared test calc used by both split criterion
		"""
		result = 0
		for var in card:
			p_ = p * probas[var]
			n_ = n * probas[var]
			result += (((prob_counts[var][1] - p_)**2)/p_ + ((prob_counts[var][0] - n_)**2)/n_) if probas[var] != 0.0 else 0.0

		return result > self.chi_squared_min

	def info(self, p, n):
		"""
		Helper function for info calc placed at the self level to avoid multiple calls
		for each split gain calc
		"""
		if (p + n) == 0.0:
			return 0.0

		# calc probabilities
		prob_p = (p)/(p + n)
		prob_n = (n)/(p + n)
		if prob_p == 0.0 and prob_n == 0.0:
			return 0.0
		elif prob_p == 0.0:
			return (-(prob_n) * math.log(prob_n, 2))
		elif prob_n == 0.0:
			return (-(prob_p) * math.log(prob_p, 2))
		else:
			return (-(prob_p) * math.log(prob_p, 2)) - ((prob_n) * math.log(prob_n, 2))

	def get_entropy(self, cardinality, A, y):
		# helper function for entropy criterion calc
		def entropy(card, B, y, p, n):
			probas, prob_counts = self.get_split_probs(card, B, y)

			# get the chi_squared test results
			chi_squared_result =  self.chi_squared_test(p, n, card, probas, prob_counts)

			# return the entropy and the chi_squared test value for the split
			gain = 0
			for val in card:
				n, p = prob_counts[val]
				gain += probas[val] * self.info(p, n)

			return gain, chi_squared_result

		# get the p/n counts for the chosen attribute A before the split
		p = float(np.sum(y==1))
		n = float(np.sum(y==0))
		# return a tuple which includes the entropy and chi_squared_test values
		return entropy(cardinality, A, y, p, n)


def main(binary_data, rank_file):
	"""
	Manages files and operations for decision tree creation
	"""
	# open and load csv files
	X, y = mlu.load_csv(binary_data)

	# parameters
	criterion = 'entropy'
	top = 'ALL'

	# initialization
	rank = InfoRank(criterion)

	# get the info gain of the top # of attributes
	print "Getting info rank list for top", top, "attributes using", criterion, "..."
	t0 = time()
	rank_list = rank.get_info_rank(top, X, y)
	print rank_list

	# write to rank file in csv format
	with open(rank_file, 'w+') as rf:
		for rank in rank_list:
			rf.write(str(rank[0]) + ', ' + str(rank[1]) + '\n');

	print "Finished - total time =", time()-t0


if __name__ == '__main__':
	"""
	The main function is called when tree.py is run from the command line with filename argumnets
	"""
	args = sys.argv[1:] # Get the filename parameters from the command line
	main( *args )
