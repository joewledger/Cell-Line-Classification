#Handles all plotting and visualization for the project
#This includes data matrix visualizations, Accuracy v. Threshold graphs, and AUC graphs

import DataFormatting as df
import SVM_Classification as svm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pylab import *
import os

def plot_accuracy_threshold_curve(all_contingency_lists, thresholds):
	plt.figure()
	x = thresholds
	y = [svm.model_accuracy(cont) for cont in all_contingency_lists]
	plt.plot(x,y)
	plt.xlabel("Threshold")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs. Threshold Curve")
	plt.savefig("Visualizations/Accuracy_Threshold.png")

def generate_prediction_heat_maps(contingency_list, threshold):
	plt.figure()
	plt.imshow(contingency_list,interpolation='none')
	for i,row in enumerate(contingency_list):
		for j,col in enumerate(row):
			percent = str(col * 100)
			length = min(len(percent),4)
			plt.text(float(j) - .15, float(i) + .05, percent[:length] + "%",size="12")
	plt.title("Model Predictions in Cross Validation (Threshold p < " + str(threshold) + ")")
	plt.xticks(arange(2),['Sensitive','Resistant'],rotation='horizontal')
	plt.yticks(arange(2),['Sensitive','Resistant'],rotation='vertical')
	plt.ylabel("Actual Values",size=24)
	plt.xlabel("Model Predictions",size=24)
	c_table_directory = "Visualizations/Cont_Tables"
	if not os.path.exists(c_table_directory): os.makedirs(c_table_directory)
	save_file = "p" + str(threshold)[str(threshold).find(".") + 1:] + ".png" 
	plt.savefig("Visualizations/Cont_Tables/" + save_file)

def visualize_matrix(matrix,title,x_axis,y_axis,outfile):
	plt.imshow(matrix)
	plt.title(title + "\n")
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)
	plt.gca().axes.get_xaxis().set_ticks([])
	plt.gca().axes.get_yaxis().set_ticks([])
	plt.savefig(outfile)

def visualize_all_data_matrices():
	mutation_dict = df.generate_mutation_dict()
	trimmed_mutation_dict = df.trim_dicts(mutation_dict, df.generate_ic_50_dict())[0]

	oncomap_matrix = df.generate_cell_line_mutation_matrix(mutation_dict)[0]
	visualize_matrix(oncomap_matrix, "Oncomap Cell Line Mutation Matrix (All)" + str(oncomap_matrix.shape),"Mutations","Cell Lines", "Visualizations/Oncomap_Cell_Line_Mutations_Matrix.png")

	oncomap_with_ic50_matrix = df.generate_cell_line_mutation_matrix(trimmed_mutation_dict)[0]
	visualize_matrix(oncomap_with_ic50_matrix, "Oncomap Cell Line Mutation Matrix (only with IC50 values) " + str(oncomap_with_ic50_matrix.shape), "Mutations","Cell Lines", "Visualizations/Oncomap_Cell_Line_Mutations_Matrix_IC50.png")

	expression_matrix = df.generate_cell_line_expression_matrix(0)[0]
	visualize_matrix(expression_matrix, "Gene Expression Matrix" + str(expression_matrix.shape),"Genes","Cell Lines", "Visualizations/Gene_Expression_Matrix.png") 

	expression_matrix_ic50 = df.generate_cell_line_expression_matrix(1)[0]
	visualize_matrix(expression_matrix_ic50, "Gene Expression Matrix (Only with IC50 values)" + str(expression_matrix_ic50.shape),"Genes","Cell Lines", "Visualizations/Gene_Expression_Matrix_IC50.png")

	expression_matrix_ic50_mutation = df.generate_cell_line_expression_matrix(2)[0]
	visualize_matrix(expression_matrix_ic50_mutation, "Gene Expression Matrix (Only with IC50 values and mutation data)" + str(expression_matrix_ic50_mutation.shape),"Genes","Cell Lines", "Visualizations/Gene_Expression_Matrix_IC50_mutation.png")