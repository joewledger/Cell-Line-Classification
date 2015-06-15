#This module is responsible for saving the results of all the experiments and visualizations
#Does not implement any of these, simply imports from the other modules and calls methods
import SVM_Classification as svmc
import DataFormatting as df
import Plotting as plt

all_evaluations = list()
thresholds = [.001,.01]
thresholds.extend([float(i) * .05 for i in range(1,5)])
svm = svmc.SVM_Classification("Expression", "IC_50_Data/CL_Sensitivity.txt" ,expression_filename="CCLE_Data/sample1000.res")


results_file = open("Results/Results.txt",'wb')
for i,threshold in enumerate(thresholds):
	print("Now working on threshold: " + str(i + 1))
	predictions,cell_lines = svm.cross_validate_make_predictions(5,threshold,cell_lines=True)
	evaluations = svm.cross_validate_evaluate_predictions(predictions=predictions)
	all_evaluations.append(evaluations)
	results_file.write("Cell line names:\n" + str(cell_lines) + "\n")
	results_file.write("Actual IC50 values for threshold: " + str(threshold) + "\n" + str([x[0] for x in predictions[0]]) + "\n")
	results_file.write("Model predictions for threshold: " + str(threshold) + "\n" + str([x[0] for x in predictions[1]]) + "\n")
	results_file.write("Model accuracy: " + str(svmc.model_accuracy(evaluations)))
	results_file.write("\n")
	plt.generate_prediction_heat_maps(evaluations,threshold)
results_file.close()

plt.plot_accuracy_threshold_curve(all_evaluations, thresholds)



#ic_50_filename = "IC_50_Data/CL_Sensitivity.txt"
#expression_features_filename = "CCLE_Data/CCLE_Expression_2012-09-29.res"
#expression_features_filename = "CCLE_Data/sample1000.res"
#mutation_features_filename = "CCLE_Data/CCLE_Oncomap3_2012-04-09.maf"

