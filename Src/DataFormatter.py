import pandas as pd
import os
from random import shuffle
import scipy.stats as sp
import numpy as np


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

def generate_ic50_series(ic50_filename):
    """
    Generates a pandas series with cell_lines as the labels and ic50 values as the entries
    """
    ic50_values = enumerate(open(ic50_filename,"rb"))
    ic50_dict = {str(row.split()[0]) : float(row.split()[1]) for row_num,row in ic50_values if row_num > 0}
    return pd.Series(ic50_dict)

def bin_ic50_series(ic50_series):
    """
    Bin ic50 values into "sensitive", "undetermined", or "resistant" bins based on ic50 values
    Top and bottom 20% are sensitive and resistant respectively, and the rest are undetermined
    :param ic50_series: the IC50 series to bin
    :return: the discretized ic50_series
    """
    ic50_values = sorted(list(ic50_series))
    lower_bound = ic50_values[int(float(len(ic50_values)) * .20)]
    upper_bound = ic50_values[int(float(len(ic50_values)) * .80)]
    binning_function = lambda score: 0 if score < lower_bound else (2 if score > upper_bound else 1)
    return ic50_series.apply(binning_function,convert_dtype=True)


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

def trim_undetermined_cell_lines(expression_frame,binned_ic50_series):
    """
    Removes any cell lines from an expression frame that are labeled undetermined in the ic50_series.
    Returns a tuple containing the expression_frame and ic50_series without undetermined cell lines.

    Assumes that expression_frame and binned_ic50_series have the same list of cell lines
    """
    dropped_cell_lines = [cell_line for cell_line in expression_frame.columns if binned_ic50_series[cell_line] == 1]
    return expression_frame.drop(labels=dropped_cell_lines,axis=1),binned_ic50_series.drop(labels=dropped_cell_lines)

def apply_pval_threshold(expression_frame,binned_ic50_series,threshold):
    """
    Trims the expression frame to include only the genes whose p-value falls under a certain threshold.
    P-value is determined by using a student t-test on the distribution of gene expression values for
    sensitive cell lines compared to the distribution of values for resistant cell lines.

    Assumes that expression_frame and binned_ic50_series have the same list of cell lines
    """
    sensitive_frame = expression_frame[[x for x in expression_frame.columns if binned_ic50_series[x] == 0]]
    resistant_frame = expression_frame[[x for x in expression_frame.columns if binned_ic50_series[x] == 2]]
    t_test = lambda gene : sp.ttest_ind(list(sensitive_frame.ix[gene]),list(resistant_frame.ix[gene]))[1]
    p_val_series = pd.Series({gene : t_test(gene) for gene in sensitive_frame.index})
    expression_frame['pval'] = p_val_series
    expression_frame = expression_frame[expression_frame['pval'] < threshold]
    del(expression_frame['pval'])
    return expression_frame

def generate_scikit_data_and_target(expression_frame,binned_ic50_series):
    """
    Returns a data array and a target array for use in a scikit-learn cross-validation experiment.

    Assumes that expression_frame and binned_ic50_series have the same list of cell lines.
    """
    data = np.array([list(expression_frame[cell_line]) for cell_line in expression_frame.columns])
    target = np.array([binned_ic50_series[cell_line] for cell_line in binned_ic50_series.index])
    return data,target

def generate_trimmed_thresholded_scikit_data_and_target(expression_filename,ic50_filename,threshold):
    expression_frame = generate_cell_line_expression_frame(expression_filename)
    ic50_series = generate_ic50_series(ic50_filename)
    binned_ic50_series = bin_ic50_series(ic50_series)
    expression_frame,binned_ic50_series = generate_cell_line_intersection(expression_frame,binned_ic50_series)
    trimmed_expression_frame,binned_ic50_series = trim_undetermined_cell_lines(expression_frame,binned_ic50_series)
    thresholded_expression_frame = apply_pval_threshold(trimmed_expression_frame,binned_ic50_series,threshold)
    return generate_scikit_data_and_target(thresholded_expression_frame,binned_ic50_series)
