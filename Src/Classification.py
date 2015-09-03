import DataFormatter as dfm

from sklearn import cross_validation
from sklearn import svm

"""
Cell Line Classification Project using SVMs (Support Vector Machines) and Neural Networks
Working with Elena Svenson under the direction of Dr. Mehmet Koyuturk
We have IC50 values for a bunch of different cell lines for the drug we are testing (SMAPs -- Small Molecular Activators of PP2A)
We are going to apply SVM to classify cell lines as either sensitive or resistant to this drug
The training input values are the gene expression measurements for each cell line
The training output values are the IC50 values discretized into several bins: "sensitive", "undetermined" , and "resistant"
"""

def get_neural_network_model_accuracy(expression_frame, classifier_series):
    raise NotImplementedError

def construct_svc_model(**kwargs):
    return svm.SVC(**kwargs)


def get_svm_model_accuracy(model,expression_filename,ic50_filename,threshold):
    """
	Gets the cross-validation accuracy for an SVM model with given parameters
    """
    scikit_data,scikit_target = dfm.generate_trimmed_thresholded_normalized_scikit_data_and_target(expression_filename,ic50_filename,threshold)
    return cross_validation.cross_val_score(model,scikit_data,scikit_target,cv=5)

def get_svm_model_accuracy_multiple_thresholds(model,expression_filename,ic50_filename,thresholds):
    """
    Gets the cross-validation accuracy for an SVM model given multiple thresholds.
    Returns a list of tuples: (threshold, (mean of scores for each fold, confidence interval of scores))
    """
    get_mean_and_conf_int = lambda array : (array.mean(), array.std() * 2)
    model_accuracy = lambda threshold : get_svm_model_accuracy(model,expression_filename,ic50_filename,threshold)
    accuracy = lambda threshold : (threshold, get_mean_and_conf_int(model_accuracy(threshold)))
    return [accuracy(threshold) for threshold in thresholds]


def get_svm_model_coefficients(model,expression_filename,ic50_filename,threshold):
    scikit_data,scikit_target = dfm.generate_trimmed_thresholded_normalized_scikit_data_and_target(expression_filename,ic50_filename,threshold)
    model.fit(scikit_data,scikit_target)
    return model.coef_[0]