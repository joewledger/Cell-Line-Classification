##---Cell Line Classification Project using SVM and Neural Networks
##---Working with Elena Svenson under the direction of Dr. Mehmet Koyuturk
##---We have IC50 values for a bunch of different cell lines for the drug we are testing (SMAPs -- Small Molecular Activators of PP2A)
##---We are going to apply SVM to classify cell lines as either sensitive or resistant to this drug
##---The training input values are the gene expression measurements for each cell line
##---The training output values are the IC50 values discretized into several bins: "sensitive", "undetermined" , and "resistant"

from sklearn import *
import numpy as np
import DataFormatting as df
from enum import Enum
import pandas as pd
import scipy.stats as sp

class classifiers(Enum):
	sensitive = 0
	undetermined = 1
	resistant = 2

#Does n-fold cross-validation on our SVM model.
#Saves these predictions along with the actual outputs in a tuple
#	First entry in the tuple will be the actual output
#	Second entry in the tuple will be the predicted output
#Parameters: cell_lines - list of cell lines from IC_50 Data
#Parameters: z_threshold - a value that will determine which features we use based on z-score
#Parameters: num_folds - the number of folds we will use in the cross-fold validation (usually 5)
def cross_validate_make_predictions(num_folds,threshold,**kwargs):
	data_matrix = df.generate_cell_line_expression_matrix(True)
	ic_50_dict = df.trim_dict(df.generate_ic_50_dict(),list(data_matrix.columns.values))
	cell_lines = ic_50_dict.keys()
	predictions = tuple([[],[]])
	num_samples = len(cell_lines)
	class_bin = generate_class_bin()
	for fold in range(0,num_folds - 1):
		lower_bound = int(float(fold) / float(num_folds) * float(num_samples))
		upper_bound = int(float(fold + 1) / float(num_folds) * float(num_samples))
		testing_cell_lines = cell_lines[lower_bound:upper_bound]
		training_cell_lines = cell_lines[0:lower_bound]
		training_cell_lines.extend(cell_lines[upper_bound:len(cell_lines) - 1])
		trimmed_matrix = trim_expression_features(data_matrix,ic_50_dict,threshold)
		model = generate_svm_model(training_cell_lines,trimmed_matrix, ic_50_dict,class_bin)
		for cell_line in testing_cell_lines:
			cell_line_data = generate_cell_line_data(cell_line,trimmed_matrix,ic_50_dict,class_bin)
			predictions[0].append(cell_line_data[1])
			predictions[1].append(model.predict(cell_line_data[0]))
	if('cell_lines' in kwargs and kwargs['cell_lines']):
		return predictions,cell_lines
	else:
		return predictions

#Evaluates the predictions the model makes for accuracy
#Returns a 3x3 matrix
#	Row labels as actual sensitivity values (discretized)
#	Column labels as predicted sensitivity values (discretized)
#	Each entry is the percentage of times that each event happened during cross-validation
def cross_validate_evaluate_predictions(**kwargs):
	num_folds = (kwargs['num_folds'] if 'num_folds' in kwargs else 5)
	threshold = (kwargs['threshold'] if 'threshold' in kwargs else .2)
	predictions = (kwargs['predictions'] if 'predictions' in kwargs else cross_validate_make_predictions(num_folds,threshold))
	pred = [[0.0] * 3 for x in range(0,3)]
	total = float(len(predictions[1]))
	for index, actual in enumerate(predictions[0]):
		pred[actual[0]][predictions[1][index][0]] += 1.0
	pred = np.divide(pred,total)
	return pred
	
#This method will generate a SVM classifier
#Parameters: training subset - a list of the cell_line names that we will use to get features from, as well as IC50 values
#Parameters: z_threshold - a value that will determine which features we use based on z-score
def generate_svm_model(training_subset,data_matrix,ic_50_dict,class_bin):
	training_data = create_training_data(training_subset,data_matrix,ic_50_dict,class_bin)
	model = svm.LinearSVC()
	model.fit(training_data[0],[value[0] for value in training_data[1]])
	return model

#This method will return a tuple containing the training input and output for our SVM classifier based on a list of cell_line names
#Parameters: training subset - a list of the cell_line names that we will use to get features from, as well as IC50 values
#Parameters: z_threshold - a value that will determine which features we use based on z-score
#Output: a tuple, first entry is training input, second is training output
#	Example for training input --- [[0,0],[1,1]], there are two samples, each with two features
#	Example for training output -- [0,1], there are two samples, each with a classification in range (0, n_classes - 1)
def create_training_data(training_subset,data_matrix,ic_50_dict,class_bin):
	training_input = []
	training_output = []
	for cell_line in training_subset:
		cell_line_data = generate_cell_line_data(cell_line,data_matrix,ic_50_dict, class_bin)
		training_input.append(cell_line_data[0])
		training_output.append(cell_line_data[1])
	return tuple([training_input,training_output])

#This method will return a tuple containing the feature inputs and output for a particular cell line
#The first entry in the tuple will be the feature inputs for the cell_line
#	This will consist of an array of the values for each of the features
#The second entry in the tuple will be the training output for the sample
#	This will be a value that determines which class the cell line is in ("sensitive", "undetermined", "resistant" AKA SUR)
#Parameters: cell_line is the name of the cell_line we are interested in
#Parameters: z_threshold is the z-score that will determine which features are actually selected
#Parameters: class_bin is a function that will convert a numeric IC50 value into one of three classes
#	The function should take an IC50 value and convert it to a number from 0-2 (corresponds to SUR)
def generate_cell_line_data(cell_line, data_matrix,ic_50_dict,class_bin):
	feature_inputs = list(data_matrix.get(cell_line).values)
	if(any(type(x) == np.ndarray for x in feature_inputs)): feature_inputs = [0.0] * len(data_matrix.index)
	ic_50 = ic_50_dict[cell_line]
	classifier = [class_bin(ic_50)]
	return feature_inputs, classifier

#Returns a function that classifies cell lines as either sensitive or resistant
#Looks at distribution of IC50 values
#	Top 15 Percent are Resistant -- marked 2
#	Bottom 15 Percent are Senstive -- marked 0
#	Rest are neutral -- marked 1
def generate_class_bin():
	ic_50_distribution = sorted(df.generate_ic_50_dict().values())
	lower_bound = ic_50_distribution[int(float(len(ic_50_distribution)) * .15)]
	upper_bound = ic_50_distribution[int(float(len(ic_50_distribution)) * .85)]
	return lambda score: 0 if score < lower_bound else (2 if score > upper_bound else 1)

#Trims genes from the data matrix that don't have a significant difference in gene expression between the sensitive and resistant groups
#Parameters: data_matrix - the gene expression matrix
#Parameters: ic_50_dict - 
def trim_expression_features(data_matrix, ic_50_dict,threshold):
	cb = generate_class_bin()
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
	data_matrix = data_matrix.copy().drop(labels=remove_list)
	return data_matrix