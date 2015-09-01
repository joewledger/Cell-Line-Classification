import numpy as np
import pandas as pd
import os
from random import shuffle


def generate_patients_expression_frame(patients_directory):
    """
    Generates gene x patient frame
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


def generate_cell_line_expression_frame(expression_features_filename):
    """
    Generates a gene x cell_line frame
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

def generate_ic_50_series(ic_50_filename):
    """
    Generates a pandas series with cell_lines as the labels and ic50 values as the entries
    """
    ic_50_values = enumerate(open(ic_50_filename,"rb"))
    ic_50_dict = {str(row.split()[0]) : float(row.split()[1]) for row_num,row in ic_50_values if row_num > 0}
    return pd.Series(ic_50_dict)


def shuffle_frame_columns(dataframe):
    """
    Shuffles the columns of a given dataframe.
    """
    columns = list(dataframe.columns.values)
    shuffle(columns)
    return dataframe.reindex_axis(columns,axis=1)


def normalize_expression_frame(dataframe):
    """
    Performs z-score normalization on a given gene-expression frame.
    Normalizes each gene expression value by (expression_value - mean_expression) / std_dev_expression
    """
    return pd.concat([((cell_vector - cell_vector.mean()) / cell_vector.std(ddof=0)) for cell_name,cell_vector in dataframe.T.iteritems()],axis=1).T

def generate_cell_line_intersection(expression_frame, ic50_series):
    """
    Trims both an gene expression frame and an ic50 series so they have the same cell lines
    :return: trimmed expression frame, trimmed ic50 series
    """
    cell_line_intersection = expression_frame.columns.intersection(ic50_series.index)
    return expression_frame[cell_line_intersection], ic50_series[cell_line_intersection]
