# Adam Martini
# ml_suite
# 2-4-14

def boost_data(X_train, y_train, boost_count):
	"""
	Adds 'boost_count' number of positive instances to the training set.

	@returns: X_train_, y_train_ => numpy arrays of boosted training data and its corresponding set of labels
	"""
	# the number of times to boost
	print 'Getting the positive instances...'
	boost = np.array(X_train[0])
	index = 0
	num_pos = 0
	for label in y_train:
		if label == 1:
			boost = np.vstack([boost, X_train[index]])
			num_pos += 1 # increment the count of positive instances
		index += 1
	print 'Positive index count =', num_pos

	print 'Boosting the training data set...'
	# remove the dummy row
	boost = boost[1:]
	# now boost the training data
	X_train_ = np.copy(X_train)
	count = 0
	while count < boost_count:
		X_train_ = np.vstack([X_train_, boost])
		count += 1

	print 'Boosted the training label set...'
	label_boost = np.array([1])
	i = 0
	while i < num_pos:
		label_boost = np.hstack([label_boost, [1]])
		i += 1
	# remove the dummy label
	label_boost = label_boost[1:]
	# stack the label boost_count times
	i = 0
	y_train_ = np.copy(y_train)
	while i < boost_count:
		y_train_ = np.hstack([y_train_, label_boost])
		i += 1

	print 'Ready for fitting with variables X, y'
	print 'Total pos instance boost of', str(num_pos * boost_count)
	lw('Boosting the positive instances count by ' + str(boost_count * num_pos) + " instances.")
	
	# return bosted data and labels ready for fitting
	return X_train_, y_train_