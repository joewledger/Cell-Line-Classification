import DataFormatter as dfm

from sklearn import cross_validation
from sklearn import svm
from pybrain.structure import LinearLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection

"""
Cell Line Classification Project using SVMs (Support Vector Machines) and Neural Networks
Working with Elena Svenson under the direction of Dr. Mehmet Koyuturk
We have IC50 values for a bunch of different cell lines for the drug we are testing (SMAPs -- Small Molecular Activators of PP2A)
We are going to apply SVM to classify cell lines as either sensitive or resistant to this drug
The training input values are the gene expression measurements for each cell line
The training output values are the IC50 values discretized into several bins: "sensitive", "undetermined" , and "resistant"
"""

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
    Returns a list of tuples: (threshold, (mean of scores across fold, confidence interval of scores))
    """
    get_mean_and_conf_int = lambda array : (array.mean(), array.std())
    model_accuracy = lambda threshold : get_svm_model_accuracy(model,expression_filename,ic50_filename,threshold)
    accuracy = lambda threshold : (threshold, get_mean_and_conf_int(model_accuracy(threshold)))
    return [accuracy(threshold) for threshold in thresholds]


def get_svm_model_coefficients(model,expression_filename,ic50_filename,threshold):
    scikit_data,scikit_target = dfm.generate_trimmed_thresholded_normalized_scikit_data_and_target(expression_filename,ic50_filename,threshold)
    model.fit(scikit_data,scikit_target)
    return model.coef_[0]