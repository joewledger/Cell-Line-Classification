import DataFormatter as dfm
import Cross_Validator as cross_validation

from sklearn import svm
#from sknn.mlp import Classifier, Layer
from sklearn import tree


"""
Cell Line Classification Project using SVMs (Support Vector Machines) and Neural Networks
Working with Elena Svenson under the direction of Dr. Mehmet Koyuturk
We have IC50 values for a bunch of different cell lines for the drug we are testing (SMAPs -- Small Molecular Activators of PP2A)
We are going to apply SVM to classify cell lines as either sensitive or resistant to this drug
The training input values are the gene expression measurements for each cell line
The training output values are the IC50 values discretized into several bins: "sensitive", "undetermined" , and "resistant"
"""


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

def get_svm_model_num_features_selected_and_accuracy(model,expression_file,ic50_file,threshold,num_permutations):
    """
    Gets the cross-validation accuracy for an SVM model with given parameters.
    Returns a tuple containing
        a list of num_permutations x (mean number of features selected across CV folds)
        and a list of num_permutations x (mean accuracy score across CV folds
    """
    scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=None)
    features_selected = []
    accuracy_scores = []
    for i in range(0,num_permutations):
        shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
        features, accuracy = cross_validation.cross_val_score_filter_feature_selection(model,threshold,shuffled_data,shuffled_target,cv=5)
        features_selected.append(features.mean())
        accuracy_scores.append(accuracy.mean())
    return features_selected, accuracy_scores

def get_svm_model_accuracy_multiple_thresholds(model,expression_file,ic50_file,thresholds,num_permutations):
    """
    Gets the cross-validation accuracy for an SVM model given multiple thresholds.
    Returns a dictionary mapping threshold -> a list of accuracy scores for each permutation at that threshold
    """
    features_and_accuracy = lambda threshold : get_svm_model_num_features_selected_and_accuracy(model,expression_file,ic50_file,threshold,num_permutations)
    return {threshold : features_and_accuracy(threshold) for threshold in thresholds}

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



"""
Neural Network Code
"""

def get_neural_network_model_accuracy(expression_file, ic50_file,threshold,num_permutations):
    raise NotImplementedError

def get_neural_network_patient_predictions(network,expression_file,ic50_file,patient_directory,threshold):
    raise NotImplementedError

def get_neural_network_predictions_full_dataset(network,expression_file,ic50_file,threshold):
    raise NotImplementedError