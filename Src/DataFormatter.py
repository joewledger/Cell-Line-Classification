import pandas as pd
import os
import scipy.stats as sp
import numpy as np
import random

def get_expression_scikit_data_target_for_drug(expression_file, ic50_file,drug,normalized=False,trimmed=False,threshold=None):
    expression_frame,ic50_series = get_expression_frame_and_ic50_series_for_drug(expression_file,ic50_file,drug,normalized,trimmed,threshold)
    return get_scikit_data_and_target(expression_frame,ic50_series)

def get_expression_frame_and_ic50_series_for_drug(expression_file, ic50_file,drug,normalized=False,trimmed=False,threshold=None):
    expression_frame = get_cell_line_expression_frame(expression_file)
    ic50_series = get_ic50_series_for_drug(ic50_file,drug)
    ic50_series = robust_bin_ic50_series(ic50_series)
    expression_frame,ic50_series = get_cell_line_intersection(expression_frame,ic50_series)

    if(normalized):
        expression_frame = normalize_expression_frame(expression_frame)

    if(trimmed):
        expression_frame,ic50_series = trim_undetermined_cell_lines(expression_frame,ic50_series)

    if(threshold):
        expression_frame = apply_pval_threshold(expression_frame,ic50_series,threshold)

    return expression_frame,ic50_series

def get_ic50_series_for_drug(ic50_file,drug):
    df = pd.DataFrame.from_csv(ic50_file,index_col=2,sep=",")[['Compound','IC50..uM.']]
    df = df[(df['Compound'] == drug)]
    return df["IC50..uM."]

def robust_bin_ic50_series(ic50_series):
    """
    Similar to the regular binning method, but instead checks if there is a hard maximum IC50 value and how many samples have this exact measurement.
    If the percentage is above 20%, we include all samples with the maximum value as resistant, and all others as sensitive.
    If the percentage is below 20%, we include the top 20% of samples as "resistant" and the bottom 20% as sensitive
    """
    maximum_score = max(ic50_series)
    percentage = float(len(ic50_series[ic50_series == "8"])) / float(len(ic50_series))

    binning_function = None
    if(percentage > .2):
        binning_function = lambda score: 0 if score < maximum_score else 2
    else:
        ic50_values = sorted(list(ic50_series))
        lower_bound = ic50_values[int(float(len(ic50_values)) * .20)]
        upper_bound = ic50_values[int(float(len(ic50_values)) * .80)]
        binning_function = lambda score: 0 if score < lower_bound else (2 if score > upper_bound else 1)
    return ic50_series.apply(binning_function,convert_dtype=True)

def get_cell_line_and_patient_expression_data_target_for_drug(expression_file,ic50_file,patient_directory,threshold,drug):

    """
    Returns expression data and target to train a learning model with, patient data to test with along with a list of patient identifiers.
    """

    expression_frame, ic50_series = get_expression_frame_and_ic50_series_for_drug(expression_file, ic50_file,drug,normalized=True,trimmed=True,threshold=threshold)
    patient_frame = get_patients_expression_frame(patient_directory)
    patient_frame = normalize_expression_frame(patient_frame)

    expression_frame,patient_frame = get_cell_line_and_patient_expression_gene_intersection(expression_frame,patient_frame)

    expression_data,expression_target = get_scikit_data_and_target(expression_frame,ic50_series)

    patient_identifiers,patient_data = get_patient_identifiers_and_data(patient_frame)

    return expression_data,expression_target,patient_identifiers,patient_data

def get_cell_line_and_patient_expression_data_target_top_features_for_drug(expression_file,ic50_file,patient_directory,num_features,drug):
    expression_frame, ic50_series = get_expression_frame_and_ic50_series_for_drug(expression_file, ic50_file,drug,normalized=True,trimmed=True)
    top_features = get_pval_top_n_features(expression_frame,ic50_series,num_features)
    expression_frame = expression_frame.ix[top_features]

    patient_frame = get_patients_expression_frame(patient_directory)
    patient_frame = normalize_expression_frame(patient_frame)

    expression_frame,patient_frame = get_cell_line_and_patient_expression_gene_intersection(expression_frame,patient_frame)
    expression_data,expression_target = get_scikit_data_and_target(expression_frame,ic50_series)
    patient_identifiers,patient_data = get_patient_identifiers_and_data(patient_frame)

    return expression_data,expression_target,patient_identifiers,patient_data,list(top_features)


"""
Methods to read data from files into pandas data structures.
"""

def get_cell_line_expression_frame(expression_file):
    """
    Generates a gene x cell_line frame
    """
    return pd.DataFrame.from_csv(expression_file, index_col=0, sep='\t')

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

def get_pval_top_n_features(expression_frame,binned_ic50_series,num_features):
    """
    Trims the expression_frame to only contain the top n features by p-value.
    Genes are the rows, cell lines are the columns.
    Returns a pandas index object
    """
    sensitive_frame = expression_frame[[x for x in expression_frame.columns if binned_ic50_series[x] == 0]]
    resistant_frame = expression_frame[[x for x in expression_frame.columns if binned_ic50_series[x] == 2]]
    t_test = lambda gene : sp.ttest_ind(list(sensitive_frame.ix[gene]),list(resistant_frame.ix[gene]))[1]
    pval_series = pd.Series({gene : t_test(gene) for gene in sensitive_frame.index})
    ordered_pval_series = pval_series.order(ascending=True)
    top_features = ordered_pval_series.head(num_features)
    index_intersection = expression_frame.index.intersection(top_features.index)
    return index_intersection

def get_features_below_pval_threshold(expression_frame,binned_ic50_series,threshold):
    """
    Provides a list of all features below a certain p-value threshold
    Genes are the rows, cell lines are the columns.
    Returns a pandas index object
    """
    sensitive_frame = expression_frame[[x for x in expression_frame.columns if binned_ic50_series[x] == 0]]
    resistant_frame = expression_frame[[x for x in expression_frame.columns if binned_ic50_series[x] == 2]]
    t_test = lambda gene : sp.ttest_ind(list(sensitive_frame.ix[gene]),list(resistant_frame.ix[gene]))[1]
    pval_series = pd.Series({gene : t_test(gene) for gene in sensitive_frame.index})
    trimmed_pval_series = pval_series[pval_series <= threshold]

    return trimmed_pval_series.index

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

def get_expression_frame_with_features(expression_frame,features):
    return expression_frame.ix[list(features)]

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

def get_cell_line_and_patient_expression_data_target_top_features(expression_file,ic50_file,patient_directory,num_features,trimmed=False):
    expression_frame, ic50_series = get_expression_frame_and_ic50_series(expression_file, ic50_file,normalized=True,trimmed=trimmed)
    top_features = get_pval_top_n_features(expression_frame,ic50_series,num_features)
    expression_frame = expression_frame.ix[top_features]

    patient_frame = get_patients_expression_frame(patient_directory)
    patient_frame = normalize_expression_frame(patient_frame)

    expression_frame,patient_frame = get_cell_line_and_patient_expression_gene_intersection(expression_frame,patient_frame)
    expression_data,expression_target = get_scikit_data_and_target(expression_frame,ic50_series)
    patient_identifiers,patient_data = get_patient_identifiers_and_data(patient_frame)

    return expression_data,expression_target,patient_identifiers,patient_data,list(top_features)