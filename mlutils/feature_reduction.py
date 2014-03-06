# Adam Martini
# ml_suite
# 2-4-14

import numpy as np
import csv
import sys
import math

import mlutils as mlu


def reduce_features(X, rank_list, top=None):
	# removes features based on info_gain criterion and pre-calculated gain lists
	print "Reducing features using top", top, "ranked features from", rank_list
	
	# set top to maximum number of features if no value was passes
	if top is None:
		top = X.shape[1]

	# create empty numpy array based on number of selected features
	X_ = np.empty(shape=(X.shape[0], top), dtype=np.float)

	# reduce features
	print "Before processing X.shape =", X.shape
	with open(rank_list, 'rb') as csvfile:
		rank_list = list(csv.reader(csvfile))
		count = 0
		for rank, index in rank_list:
			if count < top:
				index = int(index.strip())
				for i in xrange(X_.shape[0]):
					X_[i][count] = X[i][index]
			count += 1

	print "Finished reduction with:\nX_.shape =", X_.shape
	return X_


def main(train_file, test_file, rank_list, *amounts):

	for data_file in (train_file, test_file):
		# load data
		X, y = mlu.load_csv(data_file)

		for amount in amounts:
			# reduce the features
			X_ = reduce_features(X, rank_list, int(amount))

			# append labels
			X_ = np.hstack((X_, y))

			# write the new csv file
			print "Writing reduced data file for amount =", amount, "..."	
			dfile = data_file.replace(".csv", "-F" + str(amount) + ".csv")
			with open(dfile, 'wb') as data:
				writer = csv.writer(data, delimiter=',',quoting=csv.QUOTE_MINIMAL)
				writer.writerows(list(X_))

	print "Finished!"


if __name__ == '__main__':
	args = sys.argv[1:] # Get the filename and amounts parameters from the command line
	main( *args )