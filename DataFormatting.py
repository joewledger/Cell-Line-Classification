#File for creating and visualizing feature matrices
#There are two types of matrices we will be working with
#	1) Oncomap Mutation x Cell Line matrices
#	2) Gene Expression x Cell Line matrices
#All data sources for now are from CCLE

#Format for all methods that return matrices
#	All methods will return a tuple containing the following entries
#	1) The matrix
#	2) The list of names that correspond to the row entries
#	3) The list of names that correspond to the column entries
#	4) (Optional) The matrix of z-scores that correspond to the matrix

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pylab import *
import scipy.stats.mstats as sp

#Returns a tuple containing
#	1) A matrix of cell lines x genes
#	2) A list of cell line names
#	3) A list of gene names
#	4) A matrix of z-scores
#Parameter: filter_cells - boolean that tells us whether or not to filter out cell lines not in ic_50 data,
#			or all cell lines that are not in both the IC50 and mutation data
#	0 -- do nothing, 1 -- filter ic_50 only, 2 -- filter both
def generate_cell_line_expression_matrix(filter_cells):
	genes = list(row.split("\t")[0].strip() for row in open(expression_features_filename).readlines())
	genes = filter(lambda entry: (entry != "" and entry[0].isupper()), genes)[1:len(genes)]
	cell_lines = list(entry for entry in open(expression_features_filename).readline().split("\t") if entry.find("_") > 0)
	if(filter_cells == 1): cell_lines = filter(lambda a: a[0:a.find("_")] in generate_ic_50_dict().keys(), cell_lines)
	if(filter_cells == 2): cell_lines = filter(lambda a: a[0:a.find("_")] in trim_dicts(generate_ic_50_dict(),generate_mutation_dict())[0].keys(), cell_lines)
	row_count = sum(1 for row in open(expression_features_filename))
	col_count = sum(1 for col in open(expression_features_filename).readline().split("\t"))
	m = np.zeros((len(genes),len(cell_lines)))
	with open(expression_features_filename,'rb') as f:
		lines = f.readlines()
		cell_iden = lines[0].split("\t")
		row,col = (0,0)
		gene_num,cell_num = (0,0)
		while(row < row_count):
			col,cell_num = (0,0)
			gene_vector = lines[row].split("\t")
			if(gene_num + 1 > len(genes)): break
			if(gene_vector[0] == genes[gene_num]):
				while(col < col_count):
					if(cell_num + 1 > len(cell_lines)): break
					if(cell_iden[col] == cell_lines[cell_num]):
						if(len(gene_vector) > col):
							m[gene_num][cell_num] = gene_vector[col]
						cell_num += 1
					col += 1
				gene_num += 1
			row += 1
	z_scores = np.matrix([sp.zscore(row) for row in m])
	m = np.matrix(m).transpose()
	z_scores = z_scores.transpose()
	return tuple([m,cell_lines,genes,z_scores])

#Returns a tuple containing
#	1) A matrix of cell lines x Oncomap mutations
#	2) A list of cell line names
#	3) A list of mutations
#Parameters: mutation_dict is a dictionary with keys as cell lines and entries as mutations on that cell line
def generate_cell_line_mutation_matrix(mutation_dict):
	cell_line_names = set()
	all_mutations = set()
	for key in mutation_dict.keys():
		cell_line_names.add(key)
	for value in mutation_dict.values():
		for mutation in value:
			all_mutations.add(mutation)
	cell_line_names = sorted(list(cell_line_names))
	all_mutations = (sorted(list(all_mutations)))
	matrix = []
	for cell_line in cell_line_names:
		row = [0] * len(all_mutations)
		for index,mutation in enumerate(all_mutations):
			if (mutation in mutation_dict[cell_line]): row[index] = 1 
		matrix.append(row)
	return tuple([np.matrix(matrix),cell_line_names,all_mutations])

#creates a dictionary that maps cell line names to ic_50 values
def generate_ic_50_dict():
	ic_50_dict = {}
	with open(ic_50_filename,'rb') as ic_50_values:
		for row_num,row in enumerate(ic_50_values):
			fields = row.split("\t")
			if(row_num > 0): ic_50_dict[str(fields[0])] = float(fields[1])
	return ic_50_dict

#Maps a cell line name to the set of mutations associated with it in the CCLE dataset
#Returns a dictionary that maps each cell_line to a list of tuples
#	Each tuple contains a mutated gene and its corresponding mutated protein
def generate_mutation_dict():
	mutation_dict = {}
	with open(mutation_features_filename,'rb') as mutations:
		for row_num,row in enumerate(mutations):
			if(row_num == 0): continue
			fields = row.split("\t")
			cell_line = fields[15].strip()[0:fields[15].find("_")]
			gene_name = fields[0].strip()
			protein_change = fields[32].strip()
			if(cell_line in mutation_dict):
				mutation_dict[cell_line].extend([tuple([gene_name,protein_change])])
			else:
				mutation_dict[cell_line] = [tuple([gene_name,protein_change])]
	return mutation_dict

#Trims both dictionary parameters so that they contain the same set of keys
#Returns a tuple containing the two dictionaries
#Example usage: dictionaries = trim_dicts(mutation_dict,ic_50_dict) ; mutation_dict = dictionaries[0], ic_50_dict = dictionaries[1]
def trim_dicts(dict1,dict2):
	trimmed_dict_1 = dict1.copy()
	trimmed_dict_2 = dict2.copy()
	key_set_1 = dict1.keys()
	key_set_2 = dict2.keys()
	for key in key_set_1:
		if (not key in key_set_2): trimmed_dict_1.pop(key,None)
	for key in key_set_2:
		if (not key in key_set_1): trimmed_dict_2.pop(key,None)
	return tuple([trimmed_dict_1,trimmed_dict_2])

def visualize_matrix(matrix,title,x_axis,y_axis,outfile):
	plt.imshow(matrix)
	plt.title(title + "\n")
	plt.xlabel(x_axis)
	plt.ylabel(y_axis)
	plt.gca().axes.get_xaxis().set_ticks([])
	plt.gca().axes.get_yaxis().set_ticks([])
	plt.savefig(outfile)

def visualize_all():
	mutation_dict = generate_mutation_dict()
	trimmed_mutation_dict = trim_dicts(mutation_dict, generate_ic_50_dict())[0]

	oncomap_matrix = generate_cell_line_mutation_matrix(mutation_dict)[0]
	visualize_matrix(oncomap_matrix, "Oncomap Cell Line Mutation Matrix (All)" + str(oncomap_matrix.shape),"Mutations","Cell Lines", "Visualizations/Oncomap_Cell_Line_Mutations_Matrix.png")

	oncomap_with_ic50_matrix = generate_cell_line_mutation_matrix(trimmed_mutation_dict)[0]
	visualize_matrix(oncomap_with_ic50_matrix, "Oncomap Cell Line Mutation Matrix (only with IC50 values) " + str(oncomap_with_ic50_matrix.shape), "Mutations","Cell Lines", "Visualizations/Oncomap_Cell_Line_Mutations_Matrix_IC50.png")

	expression_matrix = generate_cell_line_expression_matrix(0)[0]
	visualize_matrix(expression_matrix, "Gene Expression Matrix" + str(expression_matrix.shape),"Genes","Cell Lines", "Visualizations/Gene_Expression_Matrix.png") 

	expression_matrix_ic50 = generate_cell_line_expression_matrix(1)[0]
	visualize_matrix(expression_matrix_ic50, "Gene Expression Matrix (Only with IC50 values)" + str(expression_matrix_ic50.shape),"Genes","Cell Lines", "Visualizations/Gene_Expression_Matrix_IC50.png")

	expression_matrix_ic50_mutation = generate_cell_line_expression_matrix(2)[0]
	visualize_matrix(expression_matrix_ic50_mutation, "Gene Expression Matrix (Only with IC50 values and mutation data)" + str(expression_matrix_ic50_mutation.shape),"Genes","Cell Lines", "Visualizations/Gene_Expression_Matrix_IC50_mutation.png")

ic_50_filename = "IC_50_Data/CL_Sensitivity.txt"
expression_features_filename = "CCLE_Data/CCLE_Expression_2012-09-29.res"
#expression_features_filename = "CCLE_Data/sample.res"
mutation_features_filename = "CCLE_Data/CCLE_Oncomap3_2012-04-09.maf"

visualize_all()