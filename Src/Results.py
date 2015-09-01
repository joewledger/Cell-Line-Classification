#This module is responsible for saving the results of all the experiments and visualizations
#Does not implement any of these, simply imports from the other modules and calls methods

import SVM_Classification as svmc
import DataFormatting as dfm
import Plotting as plt
import os

#Saved filenames, for convenience
ic_50_filename = "IC_50_Data/CL_Sensitivity.txt"
#expression_features_filename = "CCLE_Data/CCLE_Expression_2012-09-29.res"
expression_features_filename = "CCLE_Data/sample1000.res"
tcga_dirctory = "TCGA_Data/9f2c84a7-c887-4cb5-b6e5-d38b00d678b1/Expression-Genes/UNC__AgilentG4502A_07_3/Level_3"

all_thresholds = list()
all_accuracies = list()
all_accuracies_sensitive = list()
all_kernels = list()


def check_model_accuracy(ic50_file,expression_file):
	thresholds = generate_thresholds(.01,.20)
	for kernel in ['linear','poly','rbf']:
		svm = svmc.SVM_Classification(ic50_file,expression_file,exclude_undetermined=True,kernel=kernel,thresholds=thresholds)
		for i in range(0,10):
			all_predictions,all_features, all_evaluations = svm.evaluate_all_thresholds(5)
			accuracy_values = [svm.model_accuracy(evaluation) for evaluation in all_evaluations]
			accuracy_values_sensitive = [svm.model_accuracy_sensitive(evaluation) for evaluation in all_evaluations]
			print(kernel,thresholds, accuracy_values, accuracy_values_sensitive)
			print("\n")

def compile_all():
	reset_stored_values()
	compile_results("Tests/Linear",ic_50_filename,expression_features_filename,exclude_undetermined=True,kernel='linear',normalization=True)
	compile_results("Tests/Poly",ic_50_filename,expression_features_filename,exclude_undetermined=True,kernel='poly',normalization=True)
	compile_results("Tests/RBF",ic_50_filename,expression_features_filename,exclude_undetermined=True,kernel='rbf',normalization=True)
	plot_kernel_accuracies(all_accuracies,all_accuracies_sensitive,all_kernels,all_thresholds)


#This method saves the results of our classification experiment to a given directory
#args   -- outdir (directory to save results to)
#		-- ic50_file (file that contains ic50 data)
#		-- expression_file (file that contains expression data)
#kwargs -- model (the classification model to use, options are 'svc', 'svr', 'nn', default is 'svc')
#		-- exclude_undetermined (whether or not you would like to do training with 'undetermined' cell lines)
#		-- kernel (kernel for SVM to use. Must be one of 'linear','poly','rbf','sigmoid'. Default is 'rbf')
#		-- normalization (whether or not you want to apply normalization to the gene expression data prior to traininig the model)
#		-- num_folds (number of folds to use in cross-validation)
#		-- increment (the increment at which you want to change the threshold parameter, also used as the minimum threshold)
#		-- max_threshold (the maximum value of the threshold parameter that you would like to test)
#		-- patients (boolean indicating whether or not you want to use model to classify patients in TCGA, default False)
def compile_results(outdir,ic50_file, expression_file,**kwargs):
	print("Now working on SVM with properties: " + str(kwargs))
	outdir += "/"
	make_dirs(outdir)
	kernel = (kwargs['kernel'] if 'kernel' in kwargs else 'rbf')
	num_folds = (kwargs['num_folds'] if 'num_folds' in kwargs else 5)
	increment = (kwargs['increment'] if 'increment' in kwargs else .01)
	max_threshold = (kwargs['max_threshold'] if 'max_threshold' in kwargs else .20)
	patients = (kwargs['patients'] if 'patients' in kwargs else False)
	thresholds = generate_thresholds(increment,max_threshold)
	kwargs['thresholds'] = thresholds

	print("Now working on importing data matrices")
	df = dfm.DataFormatting(ic_50_filename ,expression_features_filename,tcga_dirctory)
	svm = svmc.SVM_Classification(ic_50_filename,expression_features_filename,**kwargs)
	cell_lines = df.generate_ic_50_dict().keys()
	print("Done importing data matrices")

	#cell_lines = df.generate_ic_50_dict().keys()
	print("Now working on Cross Validation")
	all_predictions,all_features, all_evaluations = svm.evaluate_all_thresholds(num_folds)
	accuracy_values = [svm.model_accuracy(evaluation) for evaluation in all_evaluations]
	accuracy_values_sensitive = [svm.model_accuracy_sensitive(evaluation) for evaluation in all_evaluations]
	print("Done with Cross Validation")

	print("Now working on predictions using full model")
	full_model_predictions = svm.get_all_full_model_predictions()
	print("Done making predictions using full model")
	
	write_cv_results(outdir,svm,all_predictions,all_evaluations,cell_lines,thresholds)
	write_cv_features(outdir,all_features,kernel)
	write_full_model_predictions(outdir,full_model_predictions, thresholds)
	if(kernel == 'linear'):
		write_full_model_features(outdir, full_model_predictions, thresholds,kernel)

	if(patients):
		print("Now working on patient predictions")
		patient_predictions = svm.get_all_patient_predictions()
		write_patient_results(outdir, patient_predictions,thresholds)
		print("Done working on patient predictions")
	
	plot_accuracies(outdir,accuracy_values,accuracy_values_sensitive,thresholds)
	
	all_thresholds.append(thresholds)
	all_accuracies.append(accuracy_values)
	all_accuracies_sensitive.append(accuracy_values_sensitive)
	all_kernels.append(kernel)


def write_cv_results(outdir,svm, all_predictions,all_evaluations,cell_lines, thresholds):
	cv_results_file = open(outdir + "Results/Cross-Validation-Results.txt",'wb')
	for i,evaluation in enumerate(all_evaluations):
		cv_results_file.write("Cell line names:\n%s\nActual IC50 values for threshold: %s\n%s\nModel predictions for threshold: %s\n%s\nModel accuracy: %s\n\n" % 
						  (str(cell_lines), str(thresholds[i]), str([x[0] for x in all_predictions[i][0]]), str(thresholds[i]), str([x[0] for x in all_predictions[i][1]]), str(svm.model_accuracy(evaluation))))
		plt.generate_prediction_heat_maps(outdir + "Visualizations/Cont_Tables/", evaluation,thresholds[i])
	cv_results_file.close()

def write_cv_features(outdir,all_features,kernel):
	features_file = open(outdir + "Results/Feature_Selection.txt",'wb')
	for feature in all_features:
		features_file.write("Fold: %s, Threshold: %s, Number of features: %s\nFeatures Selected: %s\n" % (str(feature[0]), str(feature[1]), str(feature[2]), str(feature[3])))
		if(kernel == 'linear'): features_file.write("Model coefficients: %s\n\n" % str([str(x) for x in feature[4]]))
	features_file.close()

def write_full_model_predictions(outdir, full_model_predictions,thresholds):
	full_model_file = open(outdir + "Results/Full_Model_Cell_Groupings.txt","wb")
	for i,prediction in enumerate(full_model_predictions):
		full_model_file.write("Threshold: %s\nCell Line Names: %s\nPredictions: %s\n\n" % (thresholds[i], str(prediction[0]), str([str(x[0]) for x in prediction[1]])))
	full_model_file.close()

def write_full_model_features(outdir, full_model_features,thresholds,kernel):
	full_model_features_file = open(outdir + "Results/Full_Model_Feature_Selection.txt","wb")
	for i,prediction in enumerate(full_model_features):
		full_model_features_file.write("Threshold: %s , Number of Features: %s\nFeatures Selected: %s\nModel coefficients: %s\n\n" % (thresholds[i], str(len(prediction[2])), str(prediction[2]),str([str(x) for x in prediction[3]])))
	full_model_features_file.close()

def write_patient_results(outdir, patient_predictions, thresholds):
	patient_file = open(outdir + "Results/Patient_Groupings.txt","wb")
	for i,prediction in enumerate(patient_predictions):
		patient_file.write("Threshold: %s\nPatient Identifiers: %s\nPredictions: %s\n\n" % (thresholds[i], str(prediction[0]), str([x[0] for x in prediction[1]])))
	patient_file.close()

def plot_accuracies(outdir,accuracy_values,accuracy_values_sensitive, thresholds):
	plt.plot_accuracy_threshold_curve(outdir + "Visualizations/Accuracy_Threshold.png",thresholds, accuracy_values)
	plt.plot_accuracy_threshold_curve(outdir + "Visualizations/Accuracy_Threshold_Sensitive.png",thresholds, accuracy_values_sensitive)

def plot_kernel_accuracies(all_accuracies,all_accuracies_sensitive,all_kernels,all_thresholds):
	plt.plot_accuracy_threshold_multiple_kernels("Tests/Kernel_Accuracy_Threshold.png",all_thresholds,all_accuracies,all_kernels)
	plt.plot_accuracy_threshold_multiple_kernels("Tests/Kernel_Accuracy_Threshold_Sensitive.png", all_thresholds,all_accuracies_sensitive,all_kernels)

def generate_thresholds(increment, max_threshold):
	return [float(i) * increment for i in range(1,int(max_threshold / increment) + 1)]

def make_dirs(outdir):
	for directory in ["","Results","Visualizations","Visualizations/Cont_Tables"]:
		out = outdir + directory
		if(not os.path.exists(out)):
			os.makedirs(out)

def reset_stored_values():
	all_thresholds = list()
	all_accuracies = list()
	all_accuracies_sensitive = list()
	all_kernels = list()

if __name__ == '__main__':
	#compile_all()
	check_model_accuracy(ic_50_filename,expression_features_filename)