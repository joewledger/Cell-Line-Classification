#Handles all plotting and visualization for the project
#This includes data matrix visualizations, Accuracy v. Threshold graphs, and AUC graphs

import matplotlib.pyplot as plt


def plot_accuracy_threshold_multiple_kernels(outfile, kernels):
    thresholds = [x[0] for x in kernels[0]]
    kernel_labels={0:'linear',1:'rbf',2:'poly'}
    kernel_colors={0:'red',1:'blue',2:'green'}
    plt.figure()
    lines = [0] * 3
    for i,kernel in enumerate(kernels):
        accuracy_scores = [x[1][0] for x in kernel]
        error_bars = [x[1][1] for x in kernel]
        lines[i], = plt.plot(thresholds,accuracy_scores,label=kernel_labels[i],color=kernel_colors[i])
        plt.errorbar(thresholds,accuracy_scores,yerr=error_bars)

    plt.legend([line for line in lines],['poly','linear','rbf'])
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Threshold Curve")
    plt.savefig(outfile)
    plt.close()


def plot_accuracy_threshold_curve(outfile, thresholds,accuracy_values):
    plt.figure()
    accuracy_scores = [x[0] for x in accuracy_values]
    error_bars = [x[1] for x in accuracy_values]
    plt.plot(thresholds, accuracy_scores)
    plt.errorbar(thresholds,accuracy_scores,yerr=error_bars)
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Threshold Curve")
    plt.savefig(outfile)
    plt.close()