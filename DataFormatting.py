#File for creating and visualizing feature matrices
#There are two types of matrices we will be working with
#	1) Oncomap Mutation x Cell Line matrices
#	2) Gene Expression x Cell Line matrices
#All data sources for now are from CCLE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pylab import *
import scipy.stats.mstats as sp
import pandas as pd
import csv

#Returns a tuple containing
#	1) A matrix of cell lines x genes
#	2) A list of cell line names
#	3) A list of gene names
#	4) A matrix of z-scores
#Parameter: filter_cells - boolean that tells us whether or not to filter out cell lines not in ic_50 data
#	True -- filter all cell lines not in IC50 data, False -- don't filter
def generate_cell_line_expression_matrix(filtering):
	df = pd.DataFrame.from_csv(expression_features_filename, index_col=0, sep='\t')
	df = df.reindex_axis(df.columns[1:], 1)
	df = df.reindex_axis([c for c in df.columns if not (c.startswith('Unnamed'))], 1)
	renamed_columns = {c: c[:c.find('_')] for c in df.columns}
	df = df.rename(columns=renamed_columns)
	df = df.drop(labels=[x for x in renamed_columns.values() if x not in generate_ic_50_dict().keys()],axis=1)
	df = df.drop(labels=[x for x in df.index if not type(x) == str or not x[0].isupper()])
	return df

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

def trim_dict(dictionary,array):
	trimmed_dict = dictionary.copy()
	for key in trimmed_dict.keys():
		if(not key in array): trimmed_dict.pop(key,None)
	return trimmed_dict

ic_50_filename = "IC_50_Data/CL_Sensitivity.txt"
#expression_features_filename = "CCLE_Data/CCLE_Expression_2012-09-29.res"
expression_features_filename = "CCLE_Data/sample.res"
mutation_features_filename = "CCLE_Data/CCLE_Oncomap3_2012-04-09.maf"
