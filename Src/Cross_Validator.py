import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel,delayed
from sklearn.utils import check_arrays
from sklearn.base import clone, is_classifier
from sklearn.metrics.scorer import _deprecate_loss_and_score_funcs
from sklearn.cross_validation import check_cv
from sklearn.cross_validation import _cross_val_score

import DataFormatter as dfm


"""
Example calls:
   cross_val_score_filter_feature_selection(model,trim_X_threshold,threshold,scikit_data,scikit_target,cv=5)
   cross_val_score_filter_feature_selection(model,trim_X_num_features,num_features,scikit_data,scikit_target,cv=5)
"""

def cross_val_score_filter_feature_selection(model,filter_function,filter_criteria, X, y=None, scoring=None, cv=None, n_jobs=1,
                    verbose=0, fit_params=None, score_func=None,
                    pre_dispatch='2*n_jobs'):
    """Evaluate a score by cross-validation

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    cv : cross-validation generator, optional, default: None
        A cross-validation generator. If None, a 3-fold cross
        validation is used or 3-fold stratified cross-validation
        when y is supplied and estimator is a classifier.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.

    verbose : integer, optional
        The verbosity level.

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    Returns
    -------
    scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
    """
    X, y = check_arrays(X, y, sparse_format='csr', allow_lists=True)
    cv = check_cv(cv, X, y, classifier=is_classifier(model))
    scorer = _deprecate_loss_and_score_funcs(
        loss_func=None,
        score_func=score_func,
        scoring=scoring
    )
    if scorer is None and not hasattr(model, 'score'):
        raise TypeError(
            "If no scoring is specified, the estimator passed "
            "should have a 'score' method. The estimator %s "
            "does not." % model)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    fit_params = fit_params if fit_params is not None else {}
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)

    scores = parallel(
        delayed(_cross_val_score)(clone(model), filter_function(X,y,train,filter_criteria), y, scorer, train, test,
                                  verbose, fit_params)
        for train, test in cv)
    return np.array(scores)

def trim_X_threshold(X,y,train,threshold):
    """
    Do calculations to trim X based on a p-value threshold
    """
    all_samples = pd.DataFrame(X)
    all_labels = pd.Series(y)

    train_samples,train_labels = get_training_samples_labels(all_samples,all_labels,train)
    features = dfm.get_features_below_pval_threshold(train_samples.T,train_labels,threshold)

    trimmed_all_samples = all_samples[features]

    return np.array([list(trimmed_all_samples.ix[row]) for row in trimmed_all_samples.index])

def trim_X_num_features(X,y,train,num_features):
    """
    Do calculations to trim X by taking the top num_features features based on p-value rank
    """
    all_samples = pd.DataFrame(X)
    all_labels = pd.Series(y)

    train_samples,train_labels = get_training_samples_labels(all_samples,all_labels,train)
    features = dfm.get_pval_top_n_features(train_samples.T,train_labels,num_features)

    trimmed_all_samples = all_samples[features]

    return np.array([list(trimmed_all_samples.ix[row]) for row in trimmed_all_samples.index])

def get_training_samples_labels(samples,labels,train):

    training_list = train.tolist()
    train_samples = samples.drop(labels=[x for x in samples.index if not training_list[x]])
    train_labels = labels.drop(labels=[x for x in labels.index if not training_list[x]])

    return train_samples,train_labels