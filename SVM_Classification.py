##---Cell Line Classification Project using SVMs (Support Vector Machines) and Neural Networks
##---Working with Elena Svenson under the direction of Dr. Mehmet Koyuturk
##---We have IC50 values for a bunch of different cell lines for the drug we are testing (SMAPs -- Small Molecular Activators of PP2A)
##---We are going to apply SVM to classify cell lines as either sensitive or resistant to this drug
##---The training input values are the gene expression measurements for each cell line
##---The training output values are the IC50 values discretized into several bins: "sensitive", "undetermined" , and "resistant"

from sklearn import *
import numpy as np
import DataFormatting as dfm
import pandas as pd
import scipy.stats as sp
import time

class SVM_Classification:

	def __init__(self,data_formatter,**kwargs):
		#Read data from files
		self.df = data_formatter
		self.model = (kwargs['model'] if 'model' in kwargs else 'svc')
		self.thresholds = (kwargs['thresholds'] if 'thresholds' in kwargs else None)
		self.exclude_undetermined = (kwargs['exclude_undetermined'] if 'exclude_undetermined' in kwargs else False)
		self.kernel = (kwargs['kernel'] if 'kernel' in kwargs else 'rbf')
		
		self.full_matrix = self.df.generate_cell_line_expression_matrix()
		self.training_matrix = self.df.strip_cell_lines_without_ic50(self.full_matrix)

		self.ic_50_dict = self.df.generate_ic_50_dict()
		self.ic_50_dict = self.df.trim_dict(self.ic_50_dict,list(self.training_matrix.columns.values))
		self.class_bin = self.generate_class_bin()
			
		#Trim data
		if(self.exclude_undetermined):
			self.training_matrix = self.training_matrix[[x for x in self.training_matrix.columns if not self.class_bin(self.ic_50_dict[x]) == 1]]
			self.ic_50_dict = self.df.trim_dict(self.ic_50_dict,list(self.training_matrix.columns.values))
		self.cell_lines = list(self.training_matrix.columns.values)
		self.num_samples = len(self.cell_lines)

	#Evaluates the model at each threshold specified in self.thresholds
	#Parameters: num_folds - the number of folds we will be using in cross-fold validation
	def evaluate_all_thresholds(self,num_folds):
		self.insignificant_gene_dict = self.generate_insignificant_genes_dict(num_folds)
		all_predictions = list()
		all_feature_selection = list()
		all_evaluations = list()
		for threshold in self.thresholds:
			prediction, feature_selction = self.cross_validate_make_predictions(num_folds,threshold)
			all_predictions.append(prediction)
			all_feature_selection.extend(feature_selction)
			evaluation = self.cross_validate_evaluate_predictions(predictions=prediction)
			all_evaluations.append(evaluation)
		return all_predictions, all_feature_selection, all_evaluations

	#Does n-fold cross-validation on our SVM model.
	#Saves these predictions along with the actual outputs in a tuple
	#	First entry in the tuple will be the actual output
	#	Second entry in the tuple will be the predicted output
	#Parameters: num_folds - the number of folds we will use in the cross-fold validation (usually 5)
	#Parameters: threshold - the specific threshold that we are using to decide which genes to include as featuress
	def cross_validate_make_predictions(self,num_folds,threshold):
		predictions = tuple([[],[]])
		feature_selection = list()
		for fold in range(0,num_folds):
			training_cell_lines, testing_cell_lines = self.split_testing_training_samples(fold,num_folds)
			data_frame = self.training_matrix.drop(labels=self.insignificant_gene_dict[(fold,threshold)])
			training_frame = data_frame[[x for x in training_cell_lines if x in data_frame.columns]]
			testing_frame = data_frame[[y for y in testing_cell_lines if y in data_frame.columns]]
			#Tuple containing fold, threshold, number of features selected, features selected, and weight of features selected
			feature_selection.append(tuple([fold,threshold,len(data_frame.index),[str(x) for x in data_frame.index],(self.get_coefficients(fold,threshold) if self.kernel == 'linear' else 0)]))
			model = self.generate_model(training_cell_lines,training_frame)
			for cell_line in testing_cell_lines:
				predictions[0].append(self.generate_cell_line_classifier(cell_line,testing_frame))
				predictions[1].append(model.predict(self.generate_cell_line_features(cell_line,testing_frame)))
		return predictions,feature_selection

	#Evaluates the predictions the model makes for accuracy
	#Returns a 2x2 matrix or a 3x3 matrix depending on whether or not we are excluding undetermined cell-lines
	#	Row labels as actual sensitivity values (discretized)
	#	Column labels as predicted sensitivity values (discretized)
	#	Each entry is the percentage of times that each event happened during cross-validation
	def cross_validate_evaluate_predictions(self,**kwargs):
		num_folds = (kwargs['num_folds'] if 'num_folds' in kwargs else 5)
		threshold = (kwargs['threshold'] if 'threshold' in kwargs else .2)
		predictions = (kwargs['predictions'] if 'predictions' in kwargs else self.cross_validate_make_predictions(num_folds,threshold)[0])
		pred = [[0.0] * 3 for x in range(0,3)]
		total = float(len(predictions[1]))
		for index, actual in enumerate(predictions[0]):
			if(self.model == 'svc'): pred[actual[0]][predictions[1][index][0]] += 1.0
			elif(self.model == 'svr'): pred[self.class_bin(actual[0])][self.class_bin(predictions[1][index][0])] += 1
		pred = np.divide(pred,total)
		if(self.exclude_undetermined):
			pred = [[pred[0][0], pred[0][2]], [pred[2][0], pred[2][2]]]
		return pred

	#This method will generate a SVM classifier
	#Parameters: training subset - a list of the cell_line names that we will use to get features from, as well as IC50 values
	#Parameters: training_matrix - a matrix containing the expression values
	def generate_model(self,training_subset,training_matrix):
		training_inputs = self.get_training_inputs(training_subset,training_matrix)
		training_outputs = self.get_training_outputs(training_subset,training_matrix)
		model = None
		if(self.model == 'svc'): model = svm.SVC(kernel=self.kernel)
		if(self.model == 'svr'): model = svm.SVR(kernel=self.kernel)
		model.fit(training_inputs,[value[0] for value in training_outputs])
		return model

	def get_training_inputs(self,training_subset,training_matrix):
		return [self.generate_cell_line_features(cell_line,training_matrix) for cell_line in training_subset]

	def get_training_outputs(self,training_subset,training_matrix):
		return [self.generate_cell_line_classifier(cell_line,training_matrix) for cell_line in training_subset]

	def generate_cell_line_features(self,cell_line,training_matrix):
		feature_inputs = list(training_matrix.ix[:,cell_line])
		if(any(type(x) == np.ndarray for x in feature_inputs) or len(feature_inputs) != len(training_matrix.index)):
			feature_inputs = [0.0] * len(training_matrix.index)
		return feature_inputs

	def generate_cell_line_classifier(self,cell_line,training_matrix):
		ic_50 = self.ic_50_dict[cell_line]
		classifier = None
		if(self.model == 'svc'): classifier = [self.class_bin(ic_50)]
		if(self.model == 'svr'): classifier = [ic_50]
		return classifier

	#Returns a function that classifies cell lines as either sensitive or resistant
	#Looks at distribution of IC50 values
	#	Top 15 Percent are Resistant -- marked 2
	#	Bottom 15 Percent are Senstive -- marked 0
	#	Rest are neutral -- marked 1
	def generate_class_bin(self):
		ic_50_distribution = sorted(self.ic_50_dict.values())
		lower_bound = ic_50_distribution[int(float(len(ic_50_distribution)) * .15)]
		upper_bound = ic_50_distribution[int(float(len(ic_50_distribution)) * .85)]
		return lambda score: 0 if score < lower_bound else (2 if score > upper_bound else 1)

	#Generates a dictionary that maps fold/threshold tuples to a list of genes that are insignificant and should be removed.
	#Parameters: num_folds, number of folds to do cross-validation with
	def generate_insignificant_genes_dict(self,num_folds):
		pval_frame = self.get_fold_gene_pvalue_frame(num_folds)
		genes_dict = {}
		for fold in range(0,num_folds):
			for threshold in self.thresholds:
				genes_dict[(fold,threshold)] = []
				for gene in pval_frame.columns:
					if(pval_frame.ix[fold, gene] > threshold):
						genes_dict[(fold,threshold)].append(gene)
		return genes_dict

	#Builds a dataframe that contains the p-values for each gene in each fold of cross-validation
	#We are precomputing these because they will be the same regardless of which threshold we are using
	#This saves a lot of computations, and it cuts the runtime by a factor of the number of thresholds we are using
	#Parameters: num_folds - the number of folds we are using in cross-validation
	def get_fold_gene_pvalue_frame(self,num_folds):
		sensitive_frame = self.training_matrix[[x for x in self.training_matrix.columns if self.class_bin(self.ic_50_dict[x]) == 0]]
		resistant_frame = self.training_matrix[[y for y in self.training_matrix.columns if self.class_bin(self.ic_50_dict[y]) == 2]]
		fold_series = []
		for fold in range(0,num_folds):
			training,testing = self.split_testing_training_samples(fold,num_folds)
			if(num_folds == 1): training = testing
			sensitive_fold = sensitive_frame[[x for x in sensitive_frame.columns if x in training]]
			resistant_fold = resistant_frame[[y for y in resistant_frame.columns if y in training]]
			fold_values = pd.Series([sp.ttest_ind(list(sensitive_fold.ix[x]),list(resistant_fold.ix[x]))[1] for x in sensitive_fold.index], index=sensitive_fold.index)
			fold_series.append(fold_values)
		return pd.DataFrame(fold_series)

	#Splits the data into testing and training samples
	#Does this for a specific fold in cross-validation, must be called repeatedly if all training and testing splits are desired
	#Parameters: fold - The fold to generate the testing and training split
	#Parameters: the number of folds we are doing cross-validation with 
	def split_testing_training_samples(self,fold, num_folds):
		lower_bound = int(float(fold) / float(num_folds) * float(self.num_samples))
		upper_bound = int(float(fold + 1) / float(num_folds) * float(self.num_samples))
		testing_cell_lines = self.cell_lines[lower_bound:upper_bound]
		training_cell_lines = self.cell_lines[0:lower_bound]
		training_cell_lines.extend(self.cell_lines[upper_bound:len(self.cell_lines) - 1])
		return training_cell_lines, testing_cell_lines

	#Determines the accuracy of the model by summing the diagonal entries in the contingency table
	#Can handle the case of a 2x2 contingency tables as well as a 3x3 contingency table
	#Parameters: contingency_list - the contingency table to evaluate
	def model_accuracy(self,contingency_list):
		return sum(contingency_list[x][x] for x in range(0,(2 if self.exclude_undetermined else 3)))

	def model_accuracy_sensitive(self,contingency_list):
		return contingency_list[0][0] / sum(contingency_list[0][x] for x in range(0,(2 if self.exclude_undetermined else 3)))

	def get_coefficients(self,fold,threshold):
		model = self.generate_model(self.cell_lines,self.training_matrix.drop(labels=self.insignificant_gene_dict[(fold,threshold)]))
		return model.coef_[0]

	#Returns a tuple containing a list of all the cell lines we are predicting and their classifications
	def get_full_model_predictions(self,threshold):
		insignificant_gene_dict = self.generate_insignificant_genes_dict(1)
		thresholded_training_matrix = self.training_matrix.drop(labels=insignificant_gene_dict[(0,threshold)])
		model = self.generate_model(self.cell_lines,thresholded_training_matrix)
		all_cell_lines = list(self.full_matrix.columns.values)
		full_features = self.get_training_inputs(all_cell_lines, self.full_matrix.drop(labels=insignificant_gene_dict[(0,threshold)]))
		return all_cell_lines, [model.predict(feature_set) for feature_set in full_features]

	#def get_patient_predictions(self,threshold):
	#	model = self.generate_model()
	#	all_cell_lines = 



