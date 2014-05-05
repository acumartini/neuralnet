# Adam Martini
# ml_suite
# 2-4-14

import csv
import numpy as np
import sys
from time import time
import datetime
import pylab as pl

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neural_network import BernoulliRBM

# sklearn utils
from sklearn.utils.extmath import density
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer

# from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as get_scores
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report as cr
from sklearn.metrics import confusion_matrix as cm

from mlutils import mlutils as mlu
from mlutils.feature_reduction import reduce_features
from mlutils.boost import boost_data


### Logging Functionaliy ###

log_file = "process_log.txt"
print 'Opening log file:', log_file
log = open(log_file, "w")
def lw(s):
	log.write(s + '\n')
def output(s):
	lw(str(s))
	print s
now = datetime.datetime.now()
lw('Log Opened @ ' + str(now.strftime("%Y-%m-%d %H:%M")) + '\n')
def lrw():
	lw(str(clfr) + '\n')
	lw(cr(y_test, y_))
	lw('\n\n')
	lw(str(cm(y_test, y_)))
	lw('\n\n')
	log.close()
	log = open(log_file, "a")


### Classification Model Processor ###

class CLFProcessor():
	def __init__(self, traindata, testdata, load_method, feature_type, boost, boost_amount, 
			     rank_list, reduce_amount):
		self.traindata = traindata
		self.testdata = testdata
		self.load_method = load_method
		self.feature_type = feature_type
		self.boost = boost
		self.boost_amount = boost_amount
		self.rank_list = rank_list
		self.reduce_amount = reduce_amount
		self.beta = 1 # f_measure

	def process(self):
		"""
		Fits and predicts using training and testing data for multiple classifiers.  All results are printed to a 
		log file and two graphs are created to show precision/recall/f_measure results and AUC.
		"""
		results = [] # stores results for each data set

		# load data
		X_train, y_train, X_test, y_test = self.clf_setup()

		# generate classification results
		### Unsueprvised ###
		# kNN
		output(80 * '=')
		output("kNN")
		results.append(self.benchmark(KNeighborsClassifier(n_neighbors=20), X_train, y_train, X_test, y_test))
		
		# NearestCentroid without threshold
		output(80 * '=')
		output("NearestCentroid (aka Rocchio classifier)")
		results.append(self.benchmark(NearestCentroid(), X_train, y_train, X_test, y_test))

		### Supervised ###
		# Logistic Regression classifiers
		for penalty in ["l2"]: #, "l1"]:
		    output(80 * '=')
		    output("LogisticRegression with %s penalty" % penalty.upper())
		    # train Logistic Regression model
		    results.append(self.benchmark(LogisticRegression(penalty=penalty, #class_weight={1:1000},
		                                  dual=False, tol=.00000001), X_train, y_train, X_test, y_test))

		# RBM/Logistic Regression Pipeline
		# output(80 * '=')
		# output("RBM/LogisticRegression Pipeline")
		# rbm = BernoulliRBM(n_components=X_train.shape[1] + 1, random_state=0, verbose=True)
		# binarizer = Binarizer(threshold=0.5)
		# X_train_b = binarizer.fit_transform(X_train, )
		# hidden_layer = rbm.fit_transform(X_train_b, y_train)
		# logistic = LogisticRegression()
		# logistic.coef_ = hidden_layer
		# # classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
		# results.append(self.benchmark(logistic, X_train, y_train, X_test, y_test))

	    # Perceptron with class weighting
		# results.append(self.benchmark(Perceptron(class_weight={0:1000}, n_iter=1000), X_train, y_train, 
		# 								X_test, y_test))
		
		# Naive Bayes classifiers
		output(80 * '=')
		print "Naive Bayes"
		lw("Naive Bayes")
		if self.feature_type == 'binary':
			results.append(self.benchmark(MultinomialNB(alpha=.01), X_train, y_train, X_test, y_test))
		else:
			results.append(self.benchmark(GaussianNB(), X_train, y_train, X_test, y_test))

		output(80 * '=')
		output("SVC with poly kernel")
		# results.append(self.benchmark(SVC(C=1000, probability=True, kernel='poly', tol=.00001), 
		# 			   X_train, y_train, X_test, y_test))
		results.append(self.benchmark(SVC(probability=True, kernel='rbf'), X_train, y_train, X_test, y_test))

		output(80 * '=')
		output("GradientBoostingClassifier with 100 estimators using entropy")
		results.append(self.benchmark(GBC(n_estimators=100), X_train, y_train, X_test, y_test))

		output(80 * '=')
		output("Random Forest Ensemble with 100 estimators using entropy")
		results.append(self.benchmark(RFC(criterion='entropy', n_estimators=100), X_train, y_train, X_test, y_test))
		
		# rearrange results into sets
		results = zip(*results)

		# create P/R and AUC graphs
		self.plot_results(results, y_test)

		output("<<< FINISHED PROCESSING >>>")
		return results, y_test

	def clf_setup(self, shuffle=False):
		# load tarining and testing data from csv
		X_train, y_train = mlu.load_csv(self.traindata, shuffle)
		X_test, y_test = mlu.load_csv(self.testdata, shuffle)

		# perform feature scaling
		X_train = mlu.scale_features(X_train, 0.0, 1.0)
		X_test = mlu.scale_features(X_test, 0.0, 1.0)

		# boost positive instances in training data
		if self.boost:
			X_train, y_train = boost_data(X_train, y_train, boost_amount)

		# perform feature reduction based on provided rank list
		if self.rank_list is not None:
			X_train(X_train, self.rank_list, self.reduce_amount)
			X_test(X_test, self.rank_list, self.reduce_amount)
			return X_train_, y_train, X_test_, y_test

		# return data sets and features for classification
		return X_train, y_train, X_test, y_test

	def benchmark(self, clf, X_train, y_train, X_test, y_test):
		output(80 * '_')

		# fit
		output("Training:")
		t0 = time()
		clf.fit(X_train, y_train)
		train_time = time() - t0
		output("train time: %0.3fs" % train_time)

		# predict
		t0 = time()
		pred = clf.predict(X_test)
		try:
			proba = clf.predict_proba(X_test)
		except:
			proba = None
		try:
			log_proba = clf.predict_log_proba(X_test)
		except:
			log_proba = None
		test_time = time() - t0
		output("test time:  %0.3fs" % test_time)

		# get metrics for the positve class only (heavy class imbalance)
		# p_score = mlu.get_pos_precision(cm(y_test, pred))
		# r_score = mlu.get_pos_recall(cm(y_test, pred))
		# f_measure = mlu.get_f_measure(p_score, r_score)

		# get metrics
		p_scores, r_scores, f_measures, support = get_scores(y_test, pred, self.beta)
		p_score_avg = p_scores.mean()
		r_score_avg = r_scores.mean()
		f_measure_avg = f_measures.mean()
		output("precision:  %0.3f \trecall:  %0.3f" % (p_score_avg, r_score_avg))

		# output results
		output("Classification results:")
		output(cr(y_test, pred))
		output(cm(y_test, pred))

		clf_descr = str(clf).split('(')[0] # get the name of the classifier from its repr()

		return clf_descr, p_score_avg, r_score_avg, f_measure_avg, train_time, test_time, proba

	def plot_results(self, results, y_test, metrics_graph=True, auc_graph=True):
		# variable setup
		reduced = self.rank_list is not None
		num_feats = self.reduce_amount

		# colors
		lllb = '#E7F5FE'
		llbb = '#D6E0FF'
		lllbb = '#EBF0FF'
		lb = '#ccffff'
		llg = '#d6fdd6'
		g = '#00E672'
		gg = '#197519'
		dg = '#005C00'
		lg = '#80CC80'
		b = '#3366FF'
		sb = '#E6FAFF'
		db = '#5548A4'
		lb = '#66CCFF'
		llb = '#99B2FF'
		y ='#FFFF00'
		r = '#E60000'
		lr = '#FF3366'
		llr = '#FF9999'

		def autolabel(rects):
			# attach some text labels
			for rect in rects:
				width = rect.get_width()
				pl.text(width+0.025, rect.get_y()+.017, '%.2f'%width,
						ha='center', va='bottom')

		# get the indices for the classifiers
		indices = np.arange(len(results[0]))

		# unpack results and average train/test times
		clf_names, p_scores, r_scores, f_measure, training_time, test_time, probas = results
		training_time = np.array(training_time) / np.max(training_time)
		test_time = np.array(test_time) / np.max(test_time)

		fig = pl.figure(figsize=(31.0, 15.0)) # was 31, 15
		fig.set_facecolor(sb)

		font = {'family' : 'sans-serif',
				'style'  : 'normal',
		        'weight' : 'semibold',
		        'size'   : 16}
		pl.rc('font', **font)

		# create precision and recall bar chart
		if (metrics_graph):
			title = pl.title("Overall Classifier Performance")
			title.set_y(1.03)
			
			rects = pl.barh(indices + .3, f_measure, .15, label=r'$\bf F_{1}$' + ' measure', color=gg)
			autolabel(rects)
			rects = pl.barh(indices + .15, r_scores, .15, label="recall", color=r)
			autolabel(rects)
			rects = pl.barh(indices, p_scores, .15, label="precision", color=b)
			autolabel(rects)
			
			pl.yticks(())
			pl.legend(loc='best', prop={'size':15})
			pl.subplots_adjust(left=.25)

			# add classifier names
			for i, c in zip(indices, clf_names):
			    pl.text(-.3, i, c)

			if self.boost:
				pl.savefig(('PRT_graph-R' + str(reduced) + '-F' + str(num_feats) + '-B' + str(self.boost_amount) + '.png'),
							facecolor=fig.get_facecolor(), edgecolor='none')
			else:
				print "Saving Metrics Plot..."
				pl.savefig(('PRT_graph-R' + str(reduced) + '-F' + str(num_feats) + '.png'),
							facecolor=fig.get_facecolor(), edgecolor='none')


		# create Proba AUC Graph
		if (auc_graph):
			prob_results = []
			for i in xrange(len(results[6])):
				if results[6][i] != None:
					prob_results.append((results[0][i], results[6][i]))
			
			pl.clf() # clear figure

			fig = pl.figure(figsize=(31.0, 15.0))
			title = pl.title("Precision and Recall Curves")
			title.set_y(1.03)

			pl.xlabel('Recall')
			ylab = pl.ylabel('Precision')
			pl.ylim([0.0, 1.05])
			pl.xlim([0.0, 1.0])

			fig.set_facecolor(llg)

			font = {'family' : 'sans-serif',
					'style'  : 'normal',
			        'weight' : 'semibold',
			        'size'   : 22}
			pl.rc('font', **font)
		
			for clf, probas in prob_results:
				if clf != 'GaussianNB': # avoid poor threshold graphing for GNB
					precision, recall, thresholds = precision_recall_curve(y_test, probas[:,1])
					area = auc(recall, precision)
					print clf, area
					pl.plot(recall, precision, label=(clf + " : AUC = " + str(area)), lw=1.3)

			pl.legend(loc='best', prop={'size':18})

			if self.boost:
				pl.savefig(('AUC_graph-R' + str(reduced) + '-F' + str(num_feats) + '-B' + str(boost_amount) + '.png'),
							facecolor=fig.get_facecolor(), edgecolor='none')
			else:
				print "Saving AUC Plot..."
				pl.savefig(('AUC_graph-R' + str(reduced) + '-F' + str(num_feats) + '.png'),
							facecolor=fig.get_facecolor(), edgecolor='none')


def main(traindata, testdata, load_method="csv", feature_type='continuous', boost=False, boost_amount=0,
	     rank_list=None, reduce_amount=None):
	# open log
	log = open(log_file, "a")
	output("<<< BEGIN PROCESSING >>>\n")
	start_str = "Parameters: " + str(boost) + ' ' + str(boost_amount) + ' ' + str(rank_list) + ' ' \
                + str(reduce_amount) + ' ' +  str(feature_type) + '\n'
	output(start_str)

	# process
	processor = CLFProcessor(traindata, testdata, load_method, feature_type, boost, boost_amount,
							 rank_list, reduce_amount)
	processor.process()

	# close log
	log.close()

if __name__ == '__main__':
	args = sys.argv[1:] # Get the filename and amounts parameters from the command line
	main( *args )
