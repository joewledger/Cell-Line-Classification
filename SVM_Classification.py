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

	def __init__(self,datatype,ic50_filename,data_file,**kwargs):
		if(datatype == "Mutation"):
			print("Not currently implemented")
		elif(datatype == "Expression"):
			#Read data from files
			self.df = dfm.DataFormatting(datatype,ic50_filename,data_file)
			self.thresholds = (kwargs['thresholds'] if 'thresholds' in kwargs else None)
			self.exclude_undetermined = (kwargs['exclude_undetermined'] if 'exclude_undetermined' in kwargs else False)
			self.data_matrix = self.df.generate_cell_line_expression_matrix(True)
			self.ic_50_dict = self.df.generate_ic_50_dict()
			self.class_bin = self.generate_class_bin()
			self.insignificant_gene_dict = None
			
			#Trim data
			if(self.exclude_undetermined):
				self.data_matrix = self.data_matrix[[x for x in self.data_matrix.columns if not self.class_bin(self.ic_50_dict[x]) == 1]]
			self.cell_lines = list(self.data_matrix.columns.values)
			self.num_samples = len(self.cell_lines)
			self.ic_50_dict = self.df.trim_dict(self.ic_50_dict,list(self.data_matrix.columns.values))


	def evaluate_all_thresholds(self,num_folds):
		self.insignificant_gene_dict = self.generate_insignificant_genes_dict(num_folds)
		all_predictions = list()
		all_feature_selection = list()
		all_evaluations = list()
		for threshold in self.thresholds:
			prediction, feature_selction = self.cross_validate_make_predictions(num_folds,threshold)
			all_predictions.append(prediction)
			all_feature_selection.append(feature_selction)
			evaluation = self.cross_validate_evaluate_predictions(predictions=prediction)
			all_evaluations.append(evaluation)
		return all_predictions, all_feature_selection, all_evaluations

	#Does n-fold cross-validation on our SVM model.
	#Saves these predictions along with the actual outputs in a tuple
	#	First entry in the tuple will be the actual output
	#	Second entry in the tuple will be the predicted output
	#Parameters: cell_lines - list of cell lines from IC_50 Data
	#Parameters: num_folds - the number of folds we will use in the cross-fold validation (usually 5)
	def cross_validate_make_predictions(self,num_folds,threshold):
		predictions = tuple([[],[]])
		feature_selection = ""
		for fold in range(0,num_folds):
			training_cell_lines, testing_cell_lines = self.split_testing_training_samples(fold,num_folds)
			data_frame = self.data_matrix.drop(labels=self.insignificant_gene_dict[(fold,threshold)])
			training_frame = data_frame[[x for x in training_cell_lines if x in data_frame.columns]]
			testing_frame = data_frame[[y for y in testing_cell_lines if y in data_frame.columns]]
			feature_selection += "Fold: " + str(fold) + ", Threshold: " + str(threshold) + ", Number of features: " + str(len(data_frame.index)) + "\n" + str(data_frame.index) + "\n"
			model = self.generate_svm_model(training_cell_lines,training_frame)
			for cell_line in testing_cell_lines:
				cell_line_data = self.generate_cell_line_data(cell_line,testing_frame)
				predictions[0].append(cell_line_data[1])
				predictions[1].append(model.predict(cell_line_data[0]))
		return predictions,feature_selection

	#Evaluates the predictions the model makes for accuracy
	#Returns a 2x2 matrix
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
			pred[actual[0]][predictions[1][index][0]] += 1.0
		pred = np.divide(pred,total)
		if(self.exclude_undetermined):
			pred = [[pred[0][0], pred[0][2]], [pred[2][0], pred[2][2]]]
		return pred

	#This method will generate a SVM classifier
	#Parameters: training subset - a list of the cell_line names that we will use to get features from, as well as IC50 values
	def generate_svm_model(self,training_subset,data_matrix):
		training_data = self.create_training_data(training_subset,data_matrix)
		model = svm.LinearSVC()
		model.fit(training_data[0],[value[0] for value in training_data[1]])
		return model

	#This method will return a tuple containing the training input and output for our SVM classifier based on a list of cell_line names
	#Parameters: training subset - a list of the cell_line names that we will use to get features from, as well as IC50 values
	#Output: a tuple, first entry is training input, second is training output
	#	Example for training input --- [[0,0],[1,1]], there are two samples, each with two features
	#	Example for training output -- [0,1], there are two samples, each with a classification in range (0, n_classes - 1)
	def create_training_data(self,training_subset,data_matrix):
		training_input = []
		training_output = []
		for cell_line in training_subset:
			cell_line_data = self.generate_cell_line_data(cell_line,data_matrix)
			training_input.append(cell_line_data[0])
			training_output.append(cell_line_data[1])
		return tuple([training_input,training_output])

	#This method will return a tuple containing the feature inputs and output for a particular cell line
	#The first entry in the tuple will be the feature inputs for the cell_line
	#	This will consist of an array of the values for each of the features
	#The second entry in the tuple will be the training output for the sample
	#	This will be a value that determines which class the cell line is in ("sensitive", "undetermined", "resistant" AKA SUR)
	#Parameters: cell_line is the name of the cell_line we are interested in
	def generate_cell_line_data(self,cell_line, data_matrix):
		feature_inputs = list(data_matrix.ix[:,cell_line])
		if(any(type(x) == np.ndarray for x in feature_inputs) or len(feature_inputs) != len(data_matrix.index)):
			feature_inputs = [0.0] * len(data_matrix.index)
		ic_50 = self.ic_50_dict[cell_line]
		classifier = [self.class_bin(ic_50)]
		return feature_inputs, classifier

	#Returns a function that classifies cell lines as either sensitive or resistant
	#Looks at distribution of IC50 values
	#	Top 15 Percent are Resistant -- marked 2
	#	Bottom 15 Percent are Senstive -- marked 0
	#	Rest are neutral -- marked 1
	def generate_class_bin(self):
		ic_50_distribution = sorted(self.df.generate_ic_50_dict().values())
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


	def get_fold_gene_pvalue_frame(self,num_folds):
		sensitive_frame = self.data_matrix[[x for x in self.data_matrix.columns if self.class_bin(self.ic_50_dict[x]) == 0]]
		resistant_frame = self.data_matrix[[y for y in self.data_matrix.columns if self.class_bin(self.ic_50_dict[y]) == 2]]
		fold_series = []
		for fold in range(0,num_folds):
			training,testing = self.split_testing_training_samples(fold,num_folds)
			sensitive_fold = sensitive_frame[[x for x in sensitive_frame.columns if x in training]]
			resistant_fold = resistant_frame[[y for y in resistant_frame.columns if y in training]]
			fold_values = pd.Series([sp.ttest_ind(list(sensitive_fold.ix[x]),list(resistant_fold.ix[x]))[1] for x in sensitive_fold.index], index=sensitive_fold.index)
			fold_series.append(fold_values)
		return pd.DataFrame(fold_series)

	def split_testing_training_samples(self,fold, num_folds):
		lower_bound = int(float(fold) / float(num_folds) * float(self.num_samples))
		upper_bound = int(float(fold + 1) / float(num_folds) * float(self.num_samples))
		testing_cell_lines = self.cell_lines[lower_bound:upper_bound]
		training_cell_lines = self.cell_lines[0:lower_bound]
		training_cell_lines.extend(self.cell_lines[upper_bound:len(self.cell_lines) - 1])
		return training_cell_lines, testing_cell_lines

	def model_accuracy(self,contingency_list):
		return sum(contingency_list[x][x] for x in range(0,(2 if self.exclude_undetermined else 3)))