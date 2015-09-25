#Handles all plotting and visualization for the project
#This includes data matrix visualizations, Accuracy v. Threshold graphs, and AUC graphs

import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_threshold_multiple_kernels(outfile, kernels):
    thresholds = sorted(kernels[0].keys())
    kernel_labels={0:'linear',1:'rbf',2:'poly'}
    kernel_colors={0:'red',1:'blue',2:'green'}
    plt.figure()
    lines = [0] * 3
    for i,kernel in enumerate(kernels):
        accuracy_means = [np.array(kernel[threshold]).mean() for threshold in thresholds]
        accuracy_std = [np.array(kernel[threshold]).std() for threshold in thresholds]
        lines[i], = plt.plot(thresholds,accuracy_means,label=kernel_labels[i],color=kernel_colors[i])
        plt.errorbar(thresholds,accuracy_means,yerr=accuracy_std)

    plt.legend([line for line in lines],['poly','linear','rbf'])
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Threshold Curve")
    plt.savefig(outfile)
    plt.close()

def plot_accuracy_num_features_multiple_kernels(outfile, kernels):
    thresholds = sorted(kernels[0].keys())
    kernel_labels={0:'linear',1:'rbf',2:'poly'}
    kernel_colors={0:'red',1:'blue',2:'green'}
    plt.figure()
    lines = [0] * 3
    for i,kernel in enumerate(kernels):
        accuracy_means = [np.array(kernel[threshold]).mean() for threshold in thresholds]
        accuracy_std = [np.array(kernel[threshold]).std() for threshold in thresholds]
        lines[i], = plt.plot(thresholds,accuracy_means,label=kernel_labels[i],color=kernel_colors[i])
        plt.errorbar(thresholds,accuracy_means,yerr=accuracy_std)

    plt.legend([line for line in lines],['poly','linear','rbf'])
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Threshold Curve")
    plt.savefig(outfile)
    plt.close()


def plot_accuracy_threshold_curve(outfile,thresholds,accuracy_scores,model_name):
    """
    Plots a threshold accuracy curve for a given model.
    :param outfile: the file to save the plot to (str)
    :param model_name: the name of the model tested (str)
    :param thresholds: the thresholds the model was tested at (list of floats)
    :param accuracy_scores: a dictionary mapping threshold -> list of accuracy scores
    """
    plt.figure()
    accuracy_means = [np.array(accuracy_scores[threshold]).mean() for threshold in thresholds]
    accuracy_std = [np.array(accuracy_scores[threshold]).std() for threshold in thresholds]
    plt.plot(thresholds, accuracy_means)
    plt.errorbar(thresholds,accuracy_means,yerr=accuracy_std)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("%s Accuracy vs. Threshold Curve" % model_name)
    plt.savefig(outfile)
    plt.close()

def plot_accuracy_num_features_curve(outfile,accuracy_scores,model_name):
    plt.figure()
    features_selected = accuracy_scores.keys()
    accuracy_means = [np.array(accuracy_scores[num_features]).mean() for num_features in features_selected]
    accuracy_std = [np.array(accuracy_scores[num_features]).std() for num_features in features_selected]

    plt.plot(features_selected, accuracy_means)
    plt.errorbar(features_selected,accuracy_means,yerr=accuracy_std)
    plt.xlabel("Number of Features Selected")
    plt.ylabel("Accuracy")
    plt.title("%s Accuracy vs. Num Features Curve" % model_name)
    plt.savefig(outfile)
    plt.close()