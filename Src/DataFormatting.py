import numpy as np
import pandas as pd
import os
from random import shuffle


def generate_patients_expression_matrix(patients_directory):
    """
    Generates gene x patient matrix
    :param patients_directory: the directory where the patient files are stored.
    """
    full_path = os.getcwd() + "/" + patients_directory
    files = [x for x in os.listdir(full_path) if x.endswith(".txt")]
    all_series = []
    for f in files:
        s = pd.Series.from_csv(full_path + "/" + f,header=-1,sep="\t")
        s.index.name = "Gene"
        s.name = s['Hybridization REF']
        s = s[2:]
        all_series.append(s)
    df = pd.DataFrame(all_series).T
    df = df.convert_objects(convert_numeric=True)
    return df


def generate_cell_line_expression_matrix(expression_features_filename):
    """
    Generates a gene x cell_line matrix
    """
    df = pd.DataFrame.from_csv(expression_features_filename, index_col=0, sep='\t')
    df = df.reindex_axis([c for c in df.columns[1:] if not c.startswith('Unnamed')], 1)
    renamed_columns = {c: c[:c.find('_')] for c in df.columns}
    df = df.rename(columns=renamed_columns)
    df = df.drop(labels=[x for x in df.index if not type(x) == str or not x[0].isupper()])
    df = df.groupby(axis=1,level=0).first()
    df = df.groupby(axis=0,level=0).first()
    df.index.name = "Genes"
    df.columns.name = "Cell_Lines"
    return df

def shuffle_matrix_columns(df):
    """
    Shuffles the columns of a given dataframe.
    """
    columns = list(df.columns.values)
    shuffle(columns)
    df = df.reindex_axis(columns,axis=1)
    return df

def normalize_expression_matrix(matrix):
    """
    Performs z-score normalization on a given gene-expression matrix.
    Normalizes each gene expression value by (expression_value - mean_expression) / std_dev_expression
    """
    return pd.concat([((cell_vector - cell_vector.mean()) / cell_vector.std(ddof=0)) for cell_name,cell_vector in matrix.T.iteritems()],axis=1).T

def strip_cell_lines_without_ic50(data_matrix):
    return data_matrix.drop(labels=[x for x in data_matrix.columns if x not in generate_ic_50_dict().keys()],axis=1)

def generate_ic_50_dict(ic_50_filename):
    """creates a dictionary that maps cell line names to ic_50 values"""
    ic_50_dict = {}
    with open(ic_50_filename,'rb') as ic_50_values:
        for row_num,row in enumerate(ic_50_values):
            fields = row.split()
            if(row_num > 0): ic_50_dict[str(fields[0])] = float(fields[1])
    return ic_50_dict