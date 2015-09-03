#Handles all plotting and visualization for the project
#This includes data matrix visualizations, Accuracy v. Threshold graphs, and AUC graphs

import DataFormatting as df
import SVM_Classification as svm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pylab import *




def plot_accuracy_threshold_multiple_kernels(outfile, thresholds, accuracy_values, kernels):
	plt.figure()
	for i, threshold in enumerate(thresholds):
		plt.plot(thresholds[i], accuracy_values[i],label=kernels[i])
	plt.legend()
	plt.xlabel("Threshold")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs. Threshold Curve")
	plt.savefig(outfile)
	plt.close()



def plot_accuracy_threshold_curve(outfile, thresholds,accuracy_values):
	plt.figure()
	plt.plot(thresholds,accuracy_values)
	plt.xlabel("Threshold")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs. Threshold Curve")
	plt.savefig(outfile)
	plt.close()