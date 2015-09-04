#Handles all plotting and visualization for the project
#This includes data matrix visualizations, Accuracy v. Threshold graphs, and AUC graphs

import matplotlib.pyplot as plt

"""
def plot_accuracy_threshold_multiple_kernels(outfile, thresholds, accuracy_values, kernels):
	plt.figure()
	for i, threshold in enumerate(thresholds):
		plt.plot(thresholds[i], accuracy_values[i],label=kernels[i])
	plt.legend()
	plt.xlabel("Threshold")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs. Threshold Curve")
	#plt.savefig(outfile)
	plt.close()
"""

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