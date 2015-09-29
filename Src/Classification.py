import DataFormatter as dfm
import Cross_Validator as cv

from sklearn import svm
#from sknn.mlp import Classifier, Layer
from sklearn import tree
from abc import ABCMeta, abstractmethod

"""
Cell Line Classification Project using SVMs (Support Vector Machines) and Neural Networks
Working with Elena Svenson under the direction of Dr. Mehmet Koyuturk
We have IC50 values for a bunch of different cell lines for the drug we are testing (SMAPs -- Small Molecular Activators of PP2A)
We are going to apply SVM to classify cell lines as either sensitive or resistant to this drug
The training input values are the gene expression measurements for each cell line
The training output values are the IC50 values discretized into several bins: "sensitive", "undetermined" , and "resistant"
"""

class Generic_Scikit_Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def construct_model(self,**kwargs):
        return None

    @abstractmethod
    def get_model_accuracy_filter_threshold(self,expression_file, ic50_file,threshold,num_permutations,**kwargs):
        model = self.construct_model(**kwargs)
        scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=threshold)
        accuracy_scores = []
        for i in range(0,num_permutations):
            shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
            accuracy_scores.append(cv.cross_val_score_filter_feature_selection(model,cv.trim_X_threshold,threshold,shuffled_data,shuffled_target,cv=5).mean())
        return accuracy_scores

    @abstractmethod
    def get_model_accuracy_filter_feature_size(self,expression_file, ic50_file,feature_size,num_permutations,**kwargs):
        model = self.construct_model(**kwargs)
        scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=None)
        accuracy_scores = []
        for i in range(0,num_permutations):
            shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
            accuracy = cross_validation.cross_val_score_filter_feature_selection(model,cross_validation.trim_X_num_features,feature_size,shuffled_data,shuffled_target,cv=5)
            accuracy_scores.append(accuracy.mean())
        return accuracy_scores

    @abstractmethod
    def get_predictions_full_CCLE_dataset(self):
        return None

    @abstractmethod
    def get_model_coefficients(self):
        return None

    @abstractmethod
    def get_patient_predictions(self):
        return None


class Decision_Tree_Model(Generic_Scikit_Model):

    def __init__(self):
        self.description = "A decision tree classification model."

    def construct_model(self,**kwargs):
        return tree.DecisionTreeClassifier(**kwargs)

    def get_model_accuracy_filter_threshold(self,expression_file, ic50_file,threshold,num_permutations,**kwargs):
        return super(Decision_Tree_Model, self).get_model_accuracy_filter_threshold(expression_file, ic50_file,threshold,num_permutations,**kwargs)

    def get_model_accuracy_filter_feature_size(self):
        return None

    def get_predictions_full_CCLE_dataset(self):
        return None

    def get_model_coefficients(self):
        return None

    def get_patient_predictions(self):
        return None



class SVM_Model(Generic_Scikit_Model):

    def __init__(self):
        self.description = "A SVM classification model."

class Neural_Network_Model(Generic_Scikit_Model):

    def __init__(self):
        self.description = "A Neural Network classification model."





"""
Decision Tree Code
"""

def construct_decision_tree_model(**kwargs):
    return tree.DecisionTreeClassifier(**kwargs)

def get_decision_tree_model_accuracy(model,expression_file, ic50_file,threshold,num_permutations):
    scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=threshold)
    accuracy_scores = []
    for i in range(0,num_permutations):
        shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
        accuracy_scores.append(cross_validation.cross_val_score_filter_feature_selection(model,shuffled_data,shuffled_target,cv=5).mean())
    return accuracy_scores

def get_decision_tree_patient_predictions(model,expression_file, ic50_file,patient_directory,threshold,trimmed=False):

    expression_data,expression_target,patient_identifiers,patient_data = dfm.get_cell_line_and_patient_expression_data_target(expression_file,ic50_file,patient_directory,threshold,trimmed=trimmed)
    model.fit(expression_data,expression_target)
    predictions = model.predict(patient_data)

    return patient_identifiers,predictions

def get_decision_tree_predictions_full_dataset(model,expression_file,ic50_file,threshold):
    training_frame,training_series = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=threshold)
    training_data,training_target = dfm.get_scikit_data_and_target(training_frame,training_series)

    cell_lines, testing_data = dfm.get_normalized_full_expression_identifiers_and_data(expression_file,training_frame.index)

    model.fit(training_data,training_target)
    predictions = model.predict(testing_data)

    return cell_lines, predictions



"""
SVM Code
"""

def construct_svc_model(**kwargs):
    return svm.SVC(**kwargs)

def get_svm_model_accuracy_for_threshold(model,expression_file,ic50_file,threshold,num_permutations):
    """
    Gets a list of the permuted cross-validation accuracies for an SVM model for a given threshold
    """
    scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=None)
    accuracy_scores = []
    for i in range(0,num_permutations):
        shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
        accuracy = cross_validation.cross_val_score_filter_feature_selection(model,cross_validation.trim_X_threshold,threshold,shuffled_data,shuffled_target,cv=5)
        accuracy_scores.append(accuracy.mean())
    return accuracy_scores

def get_svm_model_accuracy_multiple_thresholds(model,expression_file,ic50_file,thresholds,num_permutations):
    """
    Gets the cross-validation accuracy for an SVM model given multiple thresholds.
    Returns a dictionary mapping threshold -> a list of accuracy scores for each permutation at that threshold
    """
    accuracy = lambda threshold : get_svm_model_accuracy_for_threshold(model,expression_file,ic50_file,threshold,num_permutations)
    return {threshold : accuracy(threshold) for threshold in thresholds}

def get_svm_model_accuracy_for_feature_size(model,expression_file,ic50_file,feature_size,num_permutations):
    scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=None)
    accuracy_scores = []
    for i in range(0,num_permutations):
        shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
        accuracy = cross_validation.cross_val_score_filter_feature_selection(model,cross_validation.trim_X_num_features,feature_size,shuffled_data,shuffled_target,cv=5)
        accuracy_scores.append(accuracy.mean())
    return accuracy_scores

def get_svm_model_accuracy_multiple_feature_sizes(model,expression_file,ic50_file,feature_sizes,num_permutations):
    accuracy = lambda num_features : get_svm_model_accuracy_for_feature_size(model,expression_file,ic50_file,num_features,num_permutations)
    return {feature_size : accuracy(feature_size) for feature_size in feature_sizes}

def get_svm_predictions_full_dataset(model,expression_file,ic50_file,threshold):
    """
    Trains a SVM model using the partial CCLE dataset that we have IC50 values for.
    Then uses the model to make predictons for all cell lines in the CCLE dataset.
    Returns a tuple containing the list of cell lines and their predicted sensitivity
    """
    training_frame,training_series = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=threshold)
    training_data,training_target = dfm.get_scikit_data_and_target(training_frame,training_series)

    cell_lines, testing_data = dfm.get_normalized_full_expression_identifiers_and_data(expression_file,training_frame.index)

    model.fit(training_data,training_target)
    predictions = model.predict(testing_data)

    return cell_lines, predictions

def get_svm_model_coefficients(model,expression_file,ic50_file,threshold):
    """
    Returns the model coefficients for a SVM model
    """
    scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=threshold)
    model.fit(scikit_data,scikit_target)
    return model.coef_[0]

def get_svm_patient_predictions(model,expression_file,ic50_file,patient_directory,threshold,trimmed=False):
    """
    Returns the predictions for which patients are likely to be sensitive to SMAPs and which are likely to be resistant.
    First trains a given SVM model on expression data, and then uses the trained model to predict patient outcome.

    Returns a list of patient identifiers, and a list of predictions about the patients response to a given drug.
    """
    expression_data,expression_target,patient_identifiers,patient_data = dfm.get_cell_line_and_patient_expression_data_target(expression_file,ic50_file,patient_directory,threshold,trimmed=trimmed)
    model.fit(expression_data,expression_target)

    predictions = model.predict(patient_data)

    return patient_identifiers,predictions