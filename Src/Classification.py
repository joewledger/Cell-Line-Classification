import DataFormatter as dfm

import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn import tree
from pybrain.structure import LinearLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.tools.validation import CrossValidator
from pybrain.supervised.trainers import BackpropTrainer

"""
Cell Line Classification Project using SVMs (Support Vector Machines) and Neural Networks
Working with Elena Svenson under the direction of Dr. Mehmet Koyuturk
We have IC50 values for a bunch of different cell lines for the drug we are testing (SMAPs -- Small Molecular Activators of PP2A)
We are going to apply SVM to classify cell lines as either sensitive or resistant to this drug
The training input values are the gene expression measurements for each cell line
The training output values are the IC50 values discretized into several bins: "sensitive", "undetermined" , and "resistant"
"""

def get_decision_tree_model_accuracy(model,expression_file, ic50_file,threshold):
    raise NotImplementedError

def get_decision_tree_patient_predictions(model,expression_file, ic50_file,threshold):
    raise NotImplementedError

def get_decision_tree_predictions_full_dataset(model,expression_file,ic50_file,threshold):
    raise NotImplementedError

def construct_neural_net_model(num_hidden_layers,num_inputs,num_hidden_nodes,num_outputs):
    """
    Constructs a recurrent neural network with num_hidden_layers hidden layers
    Each hidden layer consists of num_hidden_nodes.
    Each hidden layer is fully connected to the next layer.
    """

    network = FeedForwardNetwork()

    input_layer = LinearLayer(num_inputs)
    hidden_layers = [LinearLayer(num_hidden_nodes)] * num_hidden_layers
    output_layer = LinearLayer(num_outputs)

    network.addInputModule(input_layer)
    for layer in hidden_layers:
        network.addModule(layer)
    network.addOutputModule(output_layer)

    connections = FullConnection(input_layer,hidden_layers[0])
    for i,layer in enumerate(hidden_layers[:-1]):
        connections.append(FullConnection(layer, hidden_layers[i + 1]))
    connections.append(FullConnection(hidden_layers[-1], output_layer))

    for connection in connections:
        network.addConnection(connection)

    return network

def get_neural_network_model_accuracy(network, expression_file, ic50_file,threshold):

    trainer = BackpropTrainer(network)
    dataset = dfm.generate_trimmed_thresholded_normalized_pybrain_dataset(expression_file,ic50_file,threshold)
    validator = CrossValidator(trainer,dataset,n_folds=5)
    return validator.validate()

def get_neural_network_patient_predictions(network,expression_file,ic50_file,patient_directory,threshold):
    raise NotImplementedError

def get_neural_network_predictions_full_dataset(network,expression_file,ic50_file,threshold):
    raise NotImplementedError

def construct_svc_model(**kwargs):
    return svm.SVC(**kwargs)

def get_svm_model_accuracy(model,expression_filename,ic50_filename,threshold,num_permutations):
    """
	Gets the cross-validation accuracy for an SVM model with given parameters.
    Returns a list containing num_permutations accuracy scores.
    """
    scikit_data,scikit_target = dfm.generate_trimmed_thresholded_normalized_scikit_data_and_target(expression_filename,ic50_filename,threshold)
    accuracy_scores = []
    for i in range(0,num_permutations):
        shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
        accuracy_scores.append(cross_validation.cross_val_score(model,shuffled_data,shuffled_target,cv=5).mean())
    return accuracy_scores

def get_svm_model_accuracy_multiple_thresholds(model,expression_file,ic50_file,thresholds,num_permutations):
    """
    Gets the cross-validation accuracy for an SVM model given multiple thresholds.
    Returns a dictionary mapping threshold -> a list of accuracy scores for each permutation at that threshold
    """
    accuracy = lambda threshold : get_svm_model_accuracy(model,expression_file,ic50_file,threshold,num_permutations)
    return {threshold : accuracy(threshold) for threshold in thresholds}

def get_svm_predictions_full_dataset(model,expression_file,ic50_file,threshold):
    """
    Trains a SVM model using the partial CCLE dataset that we have IC50 values for.
    Then uses the model to make predictons for all cell lines in the CCLE dataset.
    Returns a tuple containing the list of cell lines and their predicted sensitivity
    """
    training_frame,training_series = dfm.generate_trimmed_thresholded_normalized_expression_frame(expression_file,ic50_file,threshold)
    training_data,training_target = dfm.generate_scikit_data_and_target(training_frame,training_series)

    cell_lines, testing_data = dfm.generate_normalized_full_expression_identifiers_and_data(expression_file,training_frame.index)

    model.fit(training_data,training_target)
    predictions = model.predict(testing_data)

    return cell_lines, predictions

def get_svm_model_coefficients(model,expression_filename,ic50_filename,threshold):
    """
    Returns the model coefficients for a SVM model
    """

    scikit_data,scikit_target = dfm.generate_trimmed_thresholded_normalized_scikit_data_and_target(expression_filename,ic50_filename,threshold)
    model.fit(scikit_data,scikit_target)
    return model.coef_[0]

def get_svm_patient_predictions(model,expression_file,ic50_file,patient_directory,threshold):
    """
    Returns the predictions for which patients are likely to be sensitive to SMAPs and which are likely to be resistant.
    First trains a given SVM model on expression data, and then uses the trained model to predict patient outcome.

    Returns a list of patient identifiers, and a list of predictions about the patients response to a given drug.
    """
    expression_data,expression_target,patient_identifiers,patient_data = dfm.generate_expression_patient_data_target(expression_file,ic50_file,patient_directory,threshold)
    model.fit(expression_data,expression_target)

    predictions = model.predict(patient_data)

    return patient_identifiers,predictions