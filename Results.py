#This module is responsible for saving the results of all the experiments and visualizations
#Does not implement any of these, simply imports from the other modules and calls methods

import SVM_Classification as svmc
import DataFormatting as dfm
import Plotting as plt


def generate_thresholds():
	thresholds = [.01]
	thresholds.extend([float(i) * .05 for i in range(1,5)])
	return thresholds

data_type = "Expression"
ic_50_filename = "IC_50_Data/CL_Sensitivity.txt"
expression_features_filename = "CCLE_Data/CCLE_Expression_2012-09-29.res"
#expression_features_filename = "CCLE_Data/sample1000.res"
mutation_features_filename = "CCLE_Data/CCLE_Oncomap3_2012-04-09.maf"

thresholds = generate_thresholds()
svm = svmc.SVM_Classification(data_type, ic_50_filename ,expression_features_filename,thresholds=thresholds)

df = dfm.DataFormatting(data_type, ic_50_filename ,expression_features_filename)
all_predictions,all_evaluations = svm.evaluate_all_thresholds(5)
cell_lines = df.generate_ic_50_dict().keys()

results_file = open("Results/Results.txt",'wb')
for i,evaluation in enumerate(all_evaluations):
	results_file.write("Cell line names:\n" + str(cell_lines) + "\n")
	results_file.write("Actual IC50 values for threshold: " + str(thresholds[i]) + "\n" + str([x[0] for x in all_predictions[i][0]]) + "\n")
	results_file.write("Model predictions for threshold: " + str(thresholds[i]) + "\n" + str([x[0] for x in all_predictions[i][1]]) + "\n")
	results_file.write("Model accuracy: " + str(svmc.model_accuracy(evaluation)))
	results_file.write("\n")
	plt.generate_prediction_heat_maps(evaluation,thresholds[i])

plt.plot_accuracy_threshold_curve(all_evaluations, thresholds)

