import pandas as pd
import os
import scipy.stats as sp
import numpy as np
import random
from pybrain.datasets import ClassificationDataSet


def generate_patients_expression_frame(patient_directory):
    """
    Generates gene x patient frame
    :param patient_directory: the directory where the patient files are stored.
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

def generate_cell_line_expression_frame(expression_file):
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

def generate_ic50_series(ic50_file):
    """
    Generates a pandas series with cell_lines as the labels and ic50 values as the entries
    """
    ic50_values = enumerate(open(ic50_file,"rb"))
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

def generate_binned_ic50_series(ic50_file):
    ic50_series = generate_ic50_series(ic50_file)
    return bin_ic50_series(ic50_series)


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

def generate_trimmed_thresholded_normalized_expression_frame(expression_file,ic50_file,threshold):
    expression_frame = generate_cell_line_expression_frame(expression_file)
    expression_frame = normalize_expression_frame(expression_frame)
    ic50_series = generate_ic50_series(ic50_file)
    binned_ic50_series = bin_ic50_series(ic50_series)
    expression_frame,binned_ic50_series = generate_cell_line_intersection(expression_frame,binned_ic50_series)
    trimmed_expression_frame,binned_ic50_series = trim_undetermined_cell_lines(expression_frame,binned_ic50_series)
    thresholded_expression_frame = apply_pval_threshold(trimmed_expression_frame,binned_ic50_series,threshold)
    return thresholded_expression_frame,binned_ic50_series


def generate_trimmed_thresholded_normalized_scikit_data_and_target(expression_file,ic50_file,threshold):
    expression_frame,ic50_series = generate_trimmed_thresholded_normalized_expression_frame(expression_file,ic50_file,threshold)
    return generate_scikit_data_and_target(expression_frame,ic50_series)

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

def generate_patient_expression_gene_intersection(patient_frame,expression_frame):
    """
    Trims a patient_expression dataframe and a cell_line expression dataframe so each contains the same set of genes.
    Returns a tuple containing the patient_expression dataframe and the cell_line expression dataframe.
    """

    gene_intersection = patient_frame.index.intersection(expression_frame.index)
    return patient_frame.ix[gene_intersection], expression_frame.ix[gene_intersection]

def generate_expression_patient_data_target(expression_file,ic50_file,patient_directory,threshold,trimmed=False):
    """
    Does all steps needed to generate the training expression data and target along with the patient data for patient stratificiation.
    Returns scikit expression data and target, along with a list of patient identifiers and scikit patient data
    Expression data and patient data are normalized and have the same set of genes.
    Expression data has been trimmed by using a p_value filter.
    """

    #Generates a patient dataframe and an expression dataframe.
    patient_frame = generate_patients_expression_frame(patient_directory)
    expression_frame = generate_cell_line_expression_frame(expression_file)

    #Trims both dataframes so the set of genes they contain is the intersection of the two gene sets.
    patient_frame,expression_frame = generate_patient_expression_gene_intersection(patient_frame,expression_frame)

    #Normalizes both dataframes by gene.
    patient_frame = normalize_expression_frame(patient_frame)
    expression_frame = normalize_expression_frame(expression_frame)

    #Removes genes from the expression frame by applying the pvalue threshold.
    ic50_series = bin_ic50_series(generate_ic50_series(ic50_file))
    if(trimmed):
        expression_frame,ic50_series = trim_undetermined_cell_lines(expression_frame,ic50_series)
    expression_frame, ic50_series = generate_cell_line_intersection(expression_frame, ic50_series)
    expression_frame = apply_pval_threshold(expression_frame,ic50_series,threshold)

    #Trims both dataframes so the set of genes they contain is the intersection of the two gene sets.
    patient_frame,expression_frame = generate_patient_expression_gene_intersection(patient_frame,expression_frame)

    #Converts the expression frame into a scikit expression data and target.
    expression_data,expression_target = generate_scikit_data_and_target(expression_frame,ic50_series)

    #Converts the patient frame into a list of patient identifiers and a scikit dataset.
    patient_identifiers,patient_data = generate_patient_identifiers_and_data(patient_frame)

    #Returns a tuple containing the expression_data,expression_target,patient_identifiers, and patient_data
    return expression_data,expression_target,patient_identifiers,patient_data

def generate_patient_identifiers_and_data(patient_frame):
    """
    Converts a patient dataframe into a list of patient identifiers and a scikit patient dataset
    """

    patient_identifiers = list(patient_frame.columns)
    patient_data = np.array([list(patient_frame[column]) for column in patient_identifiers])
    return patient_identifiers,patient_data

def generate_normalized_full_expression_identifiers_and_data(expression_file,genes):

    expression_frame = generate_cell_line_expression_frame(expression_file)
    expression_frame = normalize_expression_frame(expression_frame)
    expression_frame = expression_frame.ix[expression_frame.index.intersection(genes)]

    cell_lines = expression_frame.columns
    return cell_lines, np.array([list(expression_frame[column]) for column in expression_frame.columns])

def generate_trimmed_thresholded_normalized_pybrain_dataset(expression_file,ic50_file,threshold):
    """
    Returns a trimmed, thresholded, and normalized pyBrain dataset.
    """

    expression_frame,ic50_series = generate_trimmed_thresholded_normalized_expression_frame(expression_file,ic50_file,threshold)
    return generate_pybrain_dataset(expression_frame,ic50_series)