import Src.DataFormatting as data
import os
import math

patient_directory = "Data/TCGA_Data/9f2c84a7-c887-4cb5-b6e5-d38b00d678b1/Expression-Genes/UNC__AgilentG4502A_07_3/Level_3"
#expression_file = "Data/CCLE_Data/CCLE_Expression_2012-09-29.res"
expression_file = "Data/CCLE_Data/sample1000.res"


def test_generate_patients_matrix():
    patient_matrix = data.generate_patients_expression_matrix(patient_directory)
    num_patients = len([name for name in os.listdir(patient_directory) if os.path.isfile(patient_directory + "/" + name)])
    assert len(patient_matrix.columns) == num_patients
    pass

def test_generate_expression_matrix():
    expression_matrix = data.generate_cell_line_expression_matrix(expression_file)
    assert len(list(set(expression_matrix.index))) == len(list(expression_matrix.index))
    assert len(list(set(expression_matrix.columns))) == len(list(expression_matrix.columns))
    assert not expression_matrix.isnull().values.any()
    pass

def test_normalize_expression_matrix():
    expression_matrix = data.generate_cell_line_expression_matrix(expression_file)
    normalized_matrix = data.normalize_expression_matrix(expression_matrix)
    assert len(expression_matrix.index) == len(normalized_matrix.index)
    assert len(expression_matrix.columns) == len(normalized_matrix.columns)
    row_sums = [sum(x for x in normalized_matrix.ix[name]) for name in normalized_matrix.index]
    assert(not any(math.fabs(x) > .01 for x in row_sums))
    pass