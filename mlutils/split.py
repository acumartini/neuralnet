# Adam Martini
# ml_suite
# 2-4-14

import sys
import mlutils as mlu

def split_data(dataset, load_method="csv", split=0.5, left_path="train.csv", right_path="test.csv"):
	"""
	Split the given dataset into two parts based on the split percentage and save the resulting
	split datasets.
	"""
	# convert arguments
	split = float(split)
	
	# load dataset
	data = mlu.load_data(dataset, load_method, True, False)

	# split dataset
	m = data.shape[0]
	left_data = data[0:m * split]
	right_data = data[m * split:m]

	# save split datasets
	mlu.save_data(left_data, load_method, left_path)
	mlu.save_data(right_data, load_method, right_path)

def main(dataset, load_method="csv", split=0.5, left_path="train.csv", right_path="test.csv"):
	
	print "Spitting dataset:", dataset
	split_data(dataset, load_method, split, left_path, right_path)

if __name__ == '__main__':
	args = sys.argv[1:] # Get the filename and amounts parameters from the command line
	main( *args )