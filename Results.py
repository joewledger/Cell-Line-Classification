#This module is responsible for saving the results of all the experiments and visualizations
#Does not implement any of these, simply imports from the other modules and calls methods

import SVM_Classification as svmc
import DataFormatting as dfm
import Plotting as plt
import os


def generate_thresholds(increment, max_threshold):
	thresholds = [float(i) * increment for i in range(1,int(max_threshold / increment) + 1)]
	return thresholds

def make_dirs(outdir):
	for directory in ["","Results","Visualizations","Visualizations/Cont_Tables"]:
		out = outdir + directory
		if(not os.path.exists(out)):
			os.makedirs(out)

#def compile_results(outdir,data_type, ic_50_filename, expression_features_filename, mutation_features_filename,increment,max_threshold, exclude_undetermined, kernel_type, normalization):
#kwargs -- increment
#		-- max_threshold
#		-- exclude_undetermined
#		-- kernel_type (kernel for SVM to use. Must be one of 'linear','poly','rbf','sigmoid'. Default is 'rbf')
#		-- normalization
def compile_results(outdir,ic50_file, expression_file,**kwargs):
	increment = (kwargs['increment'] if 'increment' in kwargs else .01)
	max_threshold = (kwargs['max_threshold'] if 'max_threshold' in kwargs else .20)
	exclude_undetermined = (kwargs['exclude_undetermined'] if 'exclude_undetermined' in kwargs else False)
	kernel_type = (kwargs['kernel_type'] if 'kernel_type' in kwargs else 'rbf')
	normalization = (kwargs['normalization'] if 'normalization' in kwargs else False)
	outdir += "/"
	make_dirs(outdir)
	thresholds = generate_thresholds(increment,max_threshold)
	svm = svmc.SVM_Classification(ic_50_filename,expression_features_filename,thresholds=thresholds, exclude_undetermined=exclude_undetermined,kernel=kernel_type)
	df = dfm.DataFormatting(ic_50_filename ,expression_features_filename)
	all_predictions,all_features, all_evaluations = svm.evaluate_all_thresholds(5)
	cell_lines = df.generate_ic_50_dict().keys()
	results_file = open(outdir + "Results/Results.txt",'wb')
	features_file = open(outdir + "Results/Feature_Selection.txt",'wb')
	for i,evaluation in enumerate(all_evaluations):
		results_file.write("Cell line names:\n%s\nActual IC50 values for threshold: %s\n%s\nModel predictions for threshold: %s\n%s\nModel accuracy: %s\n\n" % 
						  (str(cell_lines), str(thresholds[i]), str([x[0] for x in all_predictions[i][0]]), str(thresholds[i]), str([x[0] for x in all_predictions[i][1]]), str(svm.model_accuracy(evaluation))))
		features_file.write(all_features[i])
		plt.generate_prediction_heat_maps(outdir + "Visualizations/Cont_Tables/", evaluation,thresholds[i])
	plt.plot_accuracy_threshold_curve(outdir + "Visualizations/Accuracy_Threshold.png",thresholds,[svm.model_accuracy(evaluation) for evaluation in all_evaluations])

data_type = "Expression"
ic_50_filename = "IC_50_Data/CL_Sensitivity.txt"
#expression_features_filename = "CCLE_Data/CCLE_Expression_2012-09-29.res"
expression_features_filename = "CCLE_Data/sample1000.res"
compile_results("Test_Linear",ic_50_filename,expression_features_filename,exclude_undetermined=True,kernel_type='linear')
compile_results("Test_Poly",ic_50_filename,expression_features_filename,exclude_undetermined=True,kernel_type='poly')
compile_results("Test_RBF",ic_50_filename,expression_features_filename,exclude_undetermined=True,kernel_type='rbf')
compile_results("Test_Sigmoid",ic_50_filename,expression_features_filename,exclude_undetermined=True,kernel_type='sigmoid')

