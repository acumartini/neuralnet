"""
=====================================================================
Parameter estimation using grid search with a nested cross-validation
=====================================================================

The classifier is optimized by "nested" cross-validation using the
:class:`sklearn.grid_search.GridSearchCV` object on a development set
that comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained model is
then measured on a dedicated evaluation set that was not used during
the model selection step.

More details on tools available for model selection can be found in the
sections on :ref:`cross_validation` and :ref:`grid_search`.

"""
print __doc__

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC

import mlutils as mlu

traindata = ""
testdata = ""

# Load data
print 'Loading training dataset...'
X_train, y_train = mlu.load_csv(traindata)
print 'Loading testing dataset...'
X_test, y_test = mlu.load_csv(testdata)

print 'Starting Coarse Grid Search'
lw('Starting Coarse Grid Search\n')

n_samples = len(X_train)

# Set the parameters by cross-validation
'''
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 
                                                  2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,
                                                  2**-2, 2**1, 0.0, 2**1, 2**2, 2**3],
                        'C': [2**-5, 2**-4, 2**-3, 2**-2, 2**1, 0.0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6,
                              2**7, 2**8, 2**9, 2**10,2**11, 2**12, 2**13, 2**14, 2**15]},
                    {'kernel': ['linear'],
                        'C': [2**-5, 2**-4, 2**-3, 2**-2, 2**1, 0.0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6,
                              2**7, 2**8, 2**9, 2**10,2**11, 2**12, 2**13, 2**14, 2**15]},
                    {'kernel': ['poly'], 'gamma': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 
                                                  2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,
                                                  2**-2, 2**1, 0.0, 2**1, 2**2, 2**3],
                        'C': [2**-5, 2**-4, 2**-3, 2**-2, 2**1, 0.0, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6,
                              2**7, 2**8, 2**9, 2**10,2**11, 2**12, 2**13, 2**14, 2**15]}]

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**-5, 2**3, 0.0, 2**1, 2**3], 
                        'C': [2**3, 2**6, 2**12, 2**15]},
                    {'kernel': ['linear'],
                        'C':  [2**3, 2**6,2**12,2**15]},
                    {'kernel': ['poly'], 'gamma': [2**-5, 2**3, 0.0, 2**1, 2**3],
                        'C':  [2**3, 2**6,2**12,2**15]}]

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2**-5, 2**3, 0.0, 2**1, 2**3], 
                        'C': [2**3, 2**6, 2**12, 2**15]}]

tuned_parameters = [{'kernel': ['poly', 'rbf'], 'gamma': [2**-15, 2**-14, 2**-13, 2**-12, 2**-11, 2**-10, 
                                                  2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3,
                                                  2**-2, 2**1, 0.0, 2**1, 2**2, 2**3],
                        'C': [2**1, 2**2, 2**3, 2**4, 2**5, 2**6,
                              2**7, 2**8, 2**9, 2**10,2**11, 2**12, 2**13, 2**14, 2**15]}]
'''

tuned_parameters = [{'gamma': [2**-15, 2**-12, 2**-9, 2**-6, 2**-3, 0.0, 2**1, 2**3],
                        'C': [2**-10, 2**-4, 2**-1, 2**1, 2**2, 2**4, 2**8, 2**13, 2**15]}]

scores = [
    ('precision', precision_score),
    ('recall', recall_score),
]

for score_name, score_func in scores:
    print "# Tuning hyper-parameters for %s" % score_name
    print
    lw("# Tuning hyper-parameters for %s" % score_name + '\n')

    clf = GridSearchCV(SVC(kernel='poly', tol=.00001), tuned_parameters, score_func=score_func)
    clf.fit(X_train, y_train)

    print "Best parameters set found on development set:"
    print
    lw("Best parameters set found on development set:\n")
    print clf.best_estimator_
    lw(str(clf.best_estimator_) + '\n')
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params)
        lw("%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params) + '\n')
    print

    print "Detailed classification report:"
    lw("Detailed classification report:\n")
    print
    print "The model is trained on the full development set."
    lw("The model is trained on the previsously loaded development set.\nThe scores are computed on the full evaluation set.\n")
    print "The scores are computed on the previously loaded evaluation set."
    print
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred), '\n\n'
    lw(classification_report(y_true, y_pred) + '\n\n')
    print cm(y_true, y_pred)
    lw(str(cm(y_true, y_pred)) + '\n\n')
    print
    print
    print 'Finished Coarse Grid Search'

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.
