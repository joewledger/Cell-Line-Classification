import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel,delayed
from sklearn.base import clone, is_classifier
from sklearn.cross_validation import check_cv
from sklearn.utils import indexable
from sklearn.metrics.scorer import check_scoring
from sklearn.cross_validation import _fit_and_score


import DataFormatter as dfm


"""
Example calls:
   cross_val_score_filter_feature_selection(model,trim_X_threshold,threshold,scikit_data,scikit_target,cv=5)
   cross_val_score_filter_feature_selection(model,trim_X_num_features,num_features,scikit_data,scikit_target,cv=5)
"""

def cross_val_score_filter_feature_selection(model,filter_function,filter_criteria, X, y=None, scoring=None, cv=None, n_jobs=1,
                    verbose=0, fit_params=None,
                    pre_dispatch='2*n_jobs'):

    X, y = indexable(X, y)

    cv = check_cv(cv, X, y, classifier=is_classifier(model))
    scorer = check_scoring(model, scoring=scoring)
    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)
    scores = parallel(delayed(_fit_and_score)(clone(model), filter_function(X,y,train,filter_criteria), y, scorer,
                                              train, test, verbose, None,
                                              fit_params)
                      for train, test in cv)
    return np.array(scores)[:, 0]

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