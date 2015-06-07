##---Cell Line Classification Project using SVM and Neural Networks
##---Working with Elena Svenson under the direction of Dr. Mehmet Koyuturk
##---We have IC50 values for a bunch of different cell lines for the drug we are testing(What is the drug called?)
##---We are going to apply SVM to classify cell lines as either sensitive or resistant to this drug
##---The training input values are the gene expression measurements for each cell line
##---The training output values are the IC50 values discretized into several bins: "sensitive", "undetermined" , and "resistant"  (not sure about undetermined)

from sklearn import *
import numpy as np
import DataFormatting as df

#This method will generate a SVM classifier
#Parameters: training subset - a list of the cell_line names that we will use to get features from, as well as IC50 values
#Parameters: z_threshold - a value that will determine which features we use based on z-score
def generate_svm_model(training_subset):
	training_data = create_training_data(training_subset)
	model = svm.SVC()
	model.fit(training_data[0],training_data[1])
	return model

#This method will return a tuple containing the training input and output for our SVM classifier based on a list of cell_line names
#Parameters: training subset - a list of the cell_line names that we will use to get features from, as well as IC50 values
#Parameters: z_threshold - a value that will determine which features we use based on z-score
#Output: a tuple, first entry is training input, second is training output
#	Example for training input --- [[0,0],[1,1]], there are two samples, each with two features
#	Example for training output -- [0,1], there are two samples, each with a classification in range (0, n_classes - 1)
def create_training_data(training_subset):
	training_input = []
	training_output = []
	for cell_line in training_subset:
		cell_line_data = generate_cell_line_data(cell_line)
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
def generate_cell_line_data(cell_line):
	feature_inputs = read_feature_inputs(cell_line)
	ic_50 = generate_ic_50_dict()[cell_line]
	class_bin = generate_class_bin()
	classifier = [class_bin(ic_50)]
	return tuple([feature_inputs, classifier])

#Does n-fold cross-validation on our SVM model.
#Saves these predictions along with the actual outputs in a tuple
#	First entry in the tuple will be the actual output
#	Second entry in the tuple will be the predicted output
#Parameters: cell_lines - list of cell lines from IC_50 Data
#Parameters: z_threshold - a value that will determine which features we use based on z-score
#Parameters: num_folds - the number of folds we will use in the cross-fold validation (usually 5)
def cross_validate_make_predictions(cell_lines,num_folds):
	predictions = tuple([[],[]])
	num_samples = len(cell_lines)
	for fold in range(0,num_folds - 1):
		lower_bound = int(float(fold) / float(num_folds) * float(num_samples))
		upper_bound = int(float(fold + 1) / float(num_folds) * float(num_samples))
		testing_cell_lines = cell_lines[lower_bound:upper_bound]
		training_cell_lines = cell_lines[0:lower_bound].extend(cell_lines[upper_bound:len(cell_lines) - 1])
		model = generate_svm_model(training_cell_lines)
		for cell_line in testing_cell_lines:
			cell_line_data = generate_cell_line_data(cell_line)
			predictions[0].append(cell_line_data[1])
			predictions[1].append(model.predict(cell_line[0]))
	return predictions

#Returns a function that classifies cell lines as either sensitive or resistant
#Looks at distribution of IC50 values
#	Top 15 Percent are Resistant -- marked 2
#	Bottom 15 Percent are Senstive -- marked 0
#	Rest are neutral -- marked 1
def generate_class_bin():
	ic_50_distribution = sorted(df.generate_ic_50_dict().values())
	lower_bound = ic_50_distribution[int(float(len(ic_50_distribution)) * 3.0 / 20.0)]
	upper_bound = ic_50_distribution[int(float(len(ic_50_distribution)) * 17.0 / 20.0)]
	return lambda score: 0 if score < lower_bound else (2 if score > upper_bound else 1)

#Returns a vector in list form of the values for each of the features of a cell line
#For now we will only be doing OncoMap, eventually we should add parameters to allow for both OncoMap and Gene Expression data
#For OncoMap, each feature is binary. Either mutation exists or mutation does not exist
def read_feature_inputs(cell_line):
	return "In progress"

#def cross_validate_evalutate_predictions():

