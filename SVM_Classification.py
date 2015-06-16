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
			self.df = dfm.DataFormatting(datatype,ic50_filename,data_file)
			self.thresholds = (kwargs['thresholds'] if 'thresholds' in kwargs else None)
			self.data_matrix = self.df.generate_cell_line_expression_matrix(True)
			self.ic_50_dict = self.df.trim_dict(self.df.generate_ic_50_dict(),list(self.data_matrix.columns.values))
			self.cell_lines = self.ic_50_dict.keys()
			self.num_samples = len(self.cell_lines)
			self.class_bin = self.generate_class_bin()

	
	#Generates a dictionary that maps fold/threshold tuples to a list of genes that are insignificant and should be removed.
	#Parameters: num_folds, number of folds to do cross-validation with
	#def generate_insignificant_genes_dict(self,num_folds):

		

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
		p_values_frame = pd.DataFrame(fold_series).T
		return p_values_frame


	def evaluate_all_thresholds(self,num_folds):
		all_predictions = list()
		all_evaluations = list()
		for threshold in self.thresholds:
			prediction = self.cross_validate_make_predictions(num_folds,threshold)
			all_predictions.append(prediction)
			evaluation = self.cross_validate_evaluate_predictions(predictions=prediction)
			all_evaluations.append(evaluation)
		return all_predictions, all_evaluations

	#Does n-fold cross-validation on our SVM model.
	#Saves these predictions along with the actual outputs in a tuple
	#	First entry in the tuple will be the actual output
	#	Second entry in the tuple will be the predicted output
	#Parameters: cell_lines - list of cell lines from IC_50 Data
	#Parameters: num_folds - the number of folds we will use in the cross-fold validation (usually 5)
	def cross_validate_make_predictions(self,num_folds,threshold,**kwargs):
		predictions = tuple([[],[]])
		for fold in range(0,num_folds):
			training_cell_lines, testing_cell_lines = self.split_testing_training_samples(fold,num_folds)
			trimmed_matrix = self.data_matrix.copy().drop(labels=[x for x in testing_cell_lines])
			trimmed_matrix = self.trim_expression_features(trimmed_matrix,self.ic_50_dict,threshold)
			print("Threshold: " + str(threshold) + ", Number of features: " + str(len(trimmed_matrix.index)))
			model = self.generate_svm_model(training_cell_lines,trimmed_matrix)
			for cell_line in testing_cell_lines:
				cell_line_data = self.generate_cell_line_data(cell_line,trimmed_matrix)
				predictions[0].append(cell_line_data[1])
				predictions[1].append(model.predict(cell_line_data[0]))
		if('cell_lines' in kwargs and kwargs['cell_lines']):
			return predictions,self.cell_lines
		else:
			return predictions

	#Evaluates the predictions the model makes for accuracy
	#Returns a 3x3 matrix
	#	Row labels as actual sensitivity values (discretized)
	#	Column labels as predicted sensitivity values (discretized)
	#	Each entry is the percentage of times that each event happened during cross-validation
	def cross_validate_evaluate_predictions(self,**kwargs):
		num_folds = (kwargs['num_folds'] if 'num_folds' in kwargs else 5)
		threshold = (kwargs['threshold'] if 'threshold' in kwargs else .2)
		predictions = (kwargs['predictions'] if 'predictions' in kwargs else self.cross_validate_make_predictions(num_folds,threshold))
		pred = [[0.0] * 3 for x in range(0,3)]
		total = float(len(predictions[1]))
		for index, actual in enumerate(predictions[0]):
			pred[actual[0]][predictions[1][index][0]] += 1.0
		pred = np.divide(pred,total)
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
	#Parameters: class_bin is a function that will convert a numeric IC50 value into one of three classes
	#	The function should take an IC50 value and convert it to a number from 0-2 (corresponds to SUR)
	def generate_cell_line_data(self,cell_line, data_matrix):
		feature_inputs = list(data_matrix.get(cell_line).values)
		if(any(type(x) == np.ndarray for x in feature_inputs)):
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

	
	#Trims genes from the data matrix that don't have a significant difference in gene expression between the sensitive and resistant groups
	#Parameters: data_matrix - the gene expression matrix
	#Parameters: ic_50_dict - 
	def trim_expression_features(self, data_matrix, ic_50_dict,threshold):
		cb = self.generate_class_bin()
		sensitive_cells = [x for x in ic_50_dict.keys() if cb(ic_50_dict[x]) == 0]
		insig_cells = [x for x in ic_50_dict.keys() if cb(ic_50_dict[x]) == 1]
		resistant_cells = [x for x in ic_50_dict.keys() if cb(ic_50_dict[x]) == 2]
		dm = data_matrix.copy().drop(insig_cells,1)
		genes = [x[0] for x in dm.iterrows()]
		sig_cells = [x[0] for x in dm.iteritems()]
		remove_list = [] 
		for gene in genes:
			if(type(dm.get_value(gene,sig_cells[0])) == np.ndarray): 
				remove_list.append(gene)
				continue
			s_list = list()
			r_list = list()
			for cell in sig_cells:
				if(cb(ic_50_dict[cell]) == 0): s_list.append(dm.get_value(gene,cell))
				elif(cb(ic_50_dict[cell]) == 2):r_list.append(dm.get_value(gene,cell))
			#Compute the t-statistic
			t_stat,p_val = sp.ttest_ind(s_list,r_list)
			if(p_val > threshold):
				remove_list.append(gene)
		#From original data_matrix (not the copied version) drop all the rows that are in the list of genes to be thrown out
		return data_matrix.drop(labels=remove_list)

	def set_thresholds(self,thresholds):
		self.thresholds = thresholds

	def split_testing_training_samples(self,fold, num_folds):
		lower_bound = int(float(fold) / float(num_folds) * float(self.num_samples))
		upper_bound = int(float(fold + 1) / float(num_folds) * float(self.num_samples))
		testing_cell_lines = self.cell_lines[lower_bound:upper_bound]
		training_cell_lines = self.cell_lines[0:lower_bound]
		training_cell_lines.extend(self.cell_lines[upper_bound:len(self.cell_lines) - 1])
		return training_cell_lines, testing_cell_lines

def model_accuracy(contingency_list):
	return contingency_list[0][0] + contingency_list[1][1] + contingency_list[2][2]