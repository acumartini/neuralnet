# Adam Martini
# Adapted from sklearn tutorial on parameter tuning for SVC using GridSearchCV
# 2-6-14

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

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def search(X, y, clf, tuned_paramters):
    """
    Performs a grid search using sklearns grid_search functionality.  The best classifier is found
    out of the given tuned parameters using the scoring metrics below.

    @parameters:
        X, y - the training set data and labels used for tuning
        clf - the sklearn style classifier
        tuned_parameters - a dictionary of parameter:test_values pairs

    @returns:
        clf_best - the given classifier tuned optimally on the training set
    """
    scores = [
            ('precision', precision_score),
            ('recall', recall_score)
    ]

    for score_name, score_func in scores:
        # tune
        print "# Tuning hyper-parameters for %s\n" % score_name
        clf = GridSearchCV(SVC(kernel='poly', tol=.00001), tuned_parameters, score_func=score_func)
        clf.fit(X, y)

        print "Best parameters set found on development set:\n"
        print clf.best_estimator_
        
        print "Grid scores on training set:\n"
        for params, mean_score, scores in clf.grid_scores_:
            print "%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params)

        print 'Finished Coarse Grid Search'

def main(data):
    # load data
    X, y = mlu.load_csv(data)

    # initialize classifier
    clf = None

    # setup parameter tuning dict
    # example
    tuned_parameters = [{
            'gamma': [2**-15, 2**-12, 2**-9, 2**-6, 2**-3, 0.0, 2**1, 2**3],
            'C': [2**-10, 2**-4, 2**-1, 2**1, 2**2, 2**4, 2**8, 2**13, 2**15]
    }]

    # perform grid search
    print "Performing coarse parameter tuning via sklearn's GridSearchCV functionality.\n"
    search(X, y, clf, tuned_paramters)

if __name__ == '__main__':
    """
    The main function is called when gridsearch.py is run from the command line with arguments.
    """
    args = sys.argv[1:] # get arguments from the command line
    main( *args )

