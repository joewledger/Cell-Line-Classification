import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import check_arrays,check_random_state, safe_mask
from sklearn.base import clone, is_classifier
from sklearn.metrics.scorer import _deprecate_loss_and_score_funcs
from sklearn.cross_validation import check_cv
from sklearn.cross_validation import _cross_val_score

def cross_val_score_filter_feature_selcetion(estimator,threshold, X, y=None, scoring=None, cv=None, n_jobs=1,
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
    cv = check_cv(cv, X, y, classifier=is_classifier(estimator))
    scorer = _deprecate_loss_and_score_funcs(
        loss_func=None,
        score_func=score_func,
        scoring=scoring
    )
    if scorer is None and not hasattr(estimator, 'score'):
        raise TypeError(
            "If no scoring is specified, the estimator passed "
            "should have a 'score' method. The estimator %s "
            "does not." % estimator)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    fit_params = fit_params if fit_params is not None else {}
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)

    scores = parallel(
        delayed(_cross_val_score)(clone(estimator), get_trimmed_X(X,y,train,threshold), y, scorer, train, test,
                                  verbose, fit_params)
        for train, test in cv)

    return np.array(scores)

def get_trimmed_X(X,y,train,threshold):
    """
    Do calculations to trim X
    """
    all_samples = pd.DataFrame(X)
    sensitive_training_samples = pd.DataFrame([sample for index,sample in enumerate(X.tolist()) if y.tolist()[index] == 0 and train.tolist()[index]])
    resistant_training_samples = pd.DataFrame([sample for index,sample in enumerate(X.tolist()) if y.tolist()[index] == 2 and train.tolist()[index]])
    t_test = lambda gene : stats.ttest_ind(list(sensitive_training_samples[gene]),list(resistant_training_samples[gene]))[1]
    drop_genes = [gene for gene in all_samples.columns if t_test(gene) > threshold]
    trimmed_dataframe = all_samples.drop(labels=drop_genes,axis=1)

    X = np.array([list(trimmed_dataframe.ix[row]) for row in trimmed_dataframe.index])
    return X