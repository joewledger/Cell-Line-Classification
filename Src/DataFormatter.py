import pandas as pd
import os
import scipy.stats as sp
import numpy as np
import random

"""
Methods to read data from files into pandas data structures.
"""

def get_cell_line_expression_frame(expression_file):
    """
    Generates a gene x cell_line frame
    """
    df = pd.DataFrame.from_csv(expression_file, index_col=0, sep='\t')
    df = df.reindex_axis([c for c in df.columns[1:] if not c.startswith('Unnamed')], 1)
    renamed_columns = {c: c[:c.find('_')] for c in df.columns}
    df = df.rename(columns=renamed_columns)
    df = df.drop(labels=[x for x in df.index if not type(x) == str or not x[0].isupper()])
    df = df.groupby(axis=1,level=0).first()
    df = df.groupby(axis=0,level=0).first()
    df.index.name = "Genes"
    df.columns.name = "Cell_Lines"
    return df

def get_patients_expression_frame(patient_directory):
    """
    Generates gene x patient frame
    Patient identifiers are the column labels
    Gene names are the row labels
    """
    full_path = os.getcwd() + "/" + patient_directory
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
    df = df.dropna()
    return df

def get_ic50_series(ic50_file):
    """
    Generates a pandas series with cell_lines as the labels and ic50 values as the entries
    """
    ic50_values = enumerate(open(ic50_file,"rb"))
    ic50_dict = {str(row.split()[0]) : float(row.split()[1]) for row_num,row in ic50_values if row_num > 0}
    return pd.Series(ic50_dict)

"""
Methods to do a single operation to a pandas datastructure.
"""

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

def normalize_expression_frame(expression_frame):
    """
    Performs z-score normalization on a given gene-expression frame.
    Normalizes each gene expression value by (expression_value - mean_expression) / std_dev_expression
    """
    return pd.concat([((cell_vector - cell_vector.mean()) / cell_vector.std(ddof=0)) for cell_name,cell_vector in expression_frame.T.iteritems()],axis=1).T

def get_cell_line_intersection(expression_frame, ic50_series):
    """
    Trims both an gene expression frame and an ic50 series so they have the same cell lines
    :return: trimmed expression frame, trimmed ic50 series
    """
    cell_line_intersection = expression_frame.columns.intersection(ic50_series.index)
    return expression_frame[cell_line_intersection], ic50_series[cell_line_intersection]

def get_cell_line_and_patient_expression_gene_intersection(expression_frame,patient_frame):
    """
    Trims a patient_expression dataframe and a cell_line expression dataframe so each contains the same set of genes.
    Returns a tuple containing the patient_expression dataframe and the cell_line expression dataframe.
    """

    gene_intersection = patient_frame.index.intersection(expression_frame.index)
    return expression_frame.ix[gene_intersection], patient_frame.ix[gene_intersection]

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
    expression_frame = expression_frame[expression_frame['pval'] <= threshold]
    del(expression_frame['pval'])
    return expression_frame

def get_scikit_data_and_target(expression_frame,ic50_series):
    """
    Returns a data array and a target array for use in a scikit-learn cross-validation experiment.

    Assumes that expression_frame and binned_ic50_series have the same list of cell lines.
    """
    data = np.array([list(expression_frame[cell_line]) for cell_line in expression_frame.columns])
    target = np.array([ic50_series[cell_line] for cell_line in ic50_series.index])
    return data,target

def shuffle_scikit_data_target(scikit_data,scikit_target):
    """
    Shuffles a scikit data and target so the cell lines appear in a different order for cross-validation.
    Preserves the pairing of scikit_data -> scikit_target
    :return:
    """
    combined = zip(scikit_data.tolist(), scikit_target.tolist())
    random.shuffle(combined)
    scikit_data[:], scikit_target[:] = zip(*combined)
    return np.array(scikit_data),np.array(scikit_target)

def get_patient_identifiers_and_data(patient_frame):
    """
    Converts a patient dataframe into a list of patient identifiers and a scikit patient dataset
    """

    patient_identifiers = list(patient_frame.columns)
    patient_data = np.array([list(patient_frame[column]) for column in patient_identifiers])
    return patient_identifiers,patient_data

"""
def get_trimmed_thresholded_normalized_expression_frame(expression_file,ic50_file,threshold):
    expression_frame = get_cell_line_expression_frame(expression_file)
    expression_frame = normalize_expression_frame(expression_frame)
    ic50_series = get_ic50_series(ic50_file)
    binned_ic50_series = bin_ic50_series(ic50_series)
    expression_frame,binned_ic50_series = get_cell_line_intersection(expression_frame,binned_ic50_series)
    trimmed_expression_frame,binned_ic50_series = trim_undetermined_cell_lines(expression_frame,binned_ic50_series)
    thresholded_expression_frame = apply_pval_threshold(trimmed_expression_frame,binned_ic50_series,threshold)
    return thresholded_expression_frame,binned_ic50_series
""" 

"""
def get_trimmed_thresholded_normalized_scikit_data_and_target(expression_file,ic50_file,threshold):
    expression_frame,ic50_series = get_trimmed_thresholded_normalized_expression_frame(expression_file,ic50_file,threshold)
    return get_scikit_data_and_target(expression_frame,ic50_series)
"""

"""
def get_expression_patient_data_target(expression_file,ic50_file,patient_directory,threshold,trimmed=False):
    ""
    Does all steps needed to get the training expression data and target along with the patient data for patient stratificiation.
    Returns scikit expression data and target, along with a list of patient identifiers and scikit patient data
    Expression data and patient data are normalized and have the same set of genes.
    Expression data has been trimmed by using a p_value filter.
    ""

    #Generates a patient dataframe and an expression dataframe.
    patient_frame = get_patients_expression_frame(patient_directory)
    expression_frame = get_cell_line_expression_frame(expression_file)

    #Trims both dataframes so the set of genes they contain is the intersection of the two gene sets.
    patient_frame,expression_frame = get_patient_expression_gene_intersection(patient_frame,expression_frame)

    #Normalizes both dataframes by gene.
    patient_frame = normalize_expression_frame(patient_frame)
    expression_frame = normalize_expression_frame(expression_frame)

    #Removes genes from the expression frame by applying the pvalue threshold.
    ic50_series = bin_ic50_series(get_ic50_series(ic50_file))
    if(trimmed):
        expression_frame,ic50_series = trim_undetermined_cell_lines(expression_frame,ic50_series)
    expression_frame, ic50_series = get_cell_line_intersection(expression_frame, ic50_series)
    expression_frame = apply_pval_threshold(expression_frame,ic50_series,threshold)

    #Trims both dataframes so the set of genes they contain is the intersection of the two gene sets.
    patient_frame,expression_frame = get_patient_expression_gene_intersection(patient_frame,expression_frame)

    #Converts the expression frame into a scikit expression data and target.
    expression_data,expression_target = get_scikit_data_and_target(expression_frame,ic50_series)

    #Converts the patient frame into a list of patient identifiers and a scikit dataset.
    patient_identifiers,patient_data = get_patient_identifiers_and_data(patient_frame)

    #Returns a tuple containing the expression_data,expression_target,patient_identifiers, and patient_data
    return expression_data,expression_target,patient_identifiers,patient_data
"""

"""
def get_trimmed_normalized_expression_frame(expression_file,ic50_file):
    ""
    Generates expression frame, normalizes,then trims undetermined cell lines.
    Returns trimmed_normalized_expression_frame, ic50_series
    ""
    expression_frame = get_cell_line_expression_frame(expression_file)
    expression_frame = normalize_expression_frame(expression_frame)
    ic50_series = bin_ic50_series(get_ic50_series(ic50_file))
    expression_frame,ic50_series = get_cell_line_intersection(expression_frame,ic50_series)
    trimmed_frame,ic50_series = trim_undetermined_cell_lines(expression_frame,ic50_series)
    return trimmed_frame,ic50_series
"""

"""
def get_trimmed_normalized_scikit_data_and_target(expression_file,ic50_file):
    expression_frame,ic50_series = get_trimmed_normalized_expression_frame(expression_file,ic50_file)
    return get_scikit_data_and_target(expression_frame,ic50_series)
"""

"""
Methods that perform multiple single step operations at once.
"""

def get_expression_frame_and_ic50_series(expression_file, ic50_file,normalized=False,trimmed=False,threshold=None):

    expression_frame = get_cell_line_expression_frame(expression_file)
    ic50_series = get_ic50_series(ic50_file)
    ic50_series = bin_ic50_series(ic50_series)
    expression_frame,ic50_series = get_cell_line_intersection(expression_frame,ic50_series)

    if(normalized):
        expression_frame = normalize_expression_frame(expression_frame)

    if(trimmed):
        expression_frame,ic50_series = trim_undetermined_cell_lines(expression_frame,ic50_series)

    if(threshold):
        expression_frame = apply_pval_threshold(expression_frame,ic50_series,threshold)

    return expression_frame,ic50_series

def get_expression_scikit_data_target(expression_file, ic50_file,normalized=False,trimmed=False,threshold=None):
    expression_frame,ic50_series = get_expression_frame_and_ic50_series(expression_file,ic50_file,normalized=normalized,trimmed=trimmed,threshold=threshold)
    return get_scikit_data_and_target(expression_frame,ic50_series)

def get_normalized_full_expression_identifiers_and_data(expression_file,genes):
    """
    Given a list of genes,
    Returns a list of cell lines from an expression_frame,
    and the scikit data representation of the expression_frame with only the genes in the parameter list
    """

    expression_frame = get_cell_line_expression_frame(expression_file)
    expression_frame = normalize_expression_frame(expression_frame)
    expression_frame = expression_frame.ix[expression_frame.index.intersection(genes)]

    cell_lines = expression_frame.columns
    return cell_lines, np.array([list(expression_frame[column]) for column in expression_frame.columns])

def get_cell_line_and_patient_expression_data_target(expression_file,ic50_file,patient_directory,threshold,trimmed=False):

    """
    Returns expression data and target to train a learning model with, patient data to test with along with a list of patient identifiers.
    """

    expression_frame, ic50_series = get_expression_frame_and_ic50_series(expression_file, ic50_file,normalized=True,trimmed=trimmed,threshold=threshold)
    patient_frame = get_patients_expression_frame(patient_directory)
    patient_frame = normalize_expression_frame(patient_frame)

    expression_frame,patient_frame = get_cell_line_and_patient_expression_gene_intersection(expression_frame,patient_frame)

    expression_data,expression_target = get_scikit_data_and_target(expression_frame,ic50_series)

    patient_identifiers,patient_data = get_patient_identifiers_and_data(patient_frame)

    return expression_data,expression_target,patient_identifiers,patient_data
