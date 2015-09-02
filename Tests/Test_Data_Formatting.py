import Src.DataFormatting as data
import os
import math
import random

patient_directory = "Data/TCGA_Data/9f2c84a7-c887-4cb5-b6e5-d38b00d678b1/Expression-Genes/UNC__AgilentG4502A_07_3/Level_3"
#expression_file = "Data/CCLE_Data/CCLE_Expression_2012-09-29.res"
expression_file = "Data/CCLE_Data/sample1000.res"
ic_50_filename = "Data/IC_50_Data/CL_Sensitivity.txt"


def test_generate_patients_frame():
    patient_frame = data.generate_patients_expression_frame(patient_directory)
    num_patients = len([name for name in os.listdir(patient_directory) if os.path.isfile(patient_directory + "/" + name)])
    assert len(patient_frame.columns) == num_patients
    pass

def test_generate_expression_frame():
    expression_frame = data.generate_cell_line_expression_frame(expression_file)
    assert len(list(set(expression_frame.index))) == len(list(expression_frame.index))
    assert len(list(set(expression_frame.columns))) == len(list(expression_frame.columns))
    assert not expression_frame.isnull().values.any()
    pass

def test_generate_ic50_series():
    ic_50_series = data.generate_ic_50_series(ic_50_filename)
    assert not ic_50_series.isnull().values.any()
    assert len(ic_50_series.index) == len(set(ic_50_series.index))

def test_normalize_expression_frame():
    expression_frame = data.generate_cell_line_expression_frame(expression_file)
    normalized_frame = data.normalize_expression_frame(expression_frame)
    assert len(expression_frame.index) == len(normalized_frame.index)
    assert len(expression_frame.columns) == len(normalized_frame.columns)
    row_sums = [sum(x for x in normalized_frame.ix[name]) for name in normalized_frame.index]
    assert(not any(math.fabs(x) > .01 for x in row_sums))
    pass

def test_shuffle_frame_columns():
    expression_frame = data.generate_cell_line_expression_frame(expression_file)
    shuffled_frame = data.shuffle_frame_columns(expression_frame.copy())
    for i in xrange(0,20):
        cell_line = random.choice(list(expression_frame.columns))
        assert all(expression_frame[cell_line] == shuffled_frame[cell_line])

def test_generate_cell_line_intersection():
    expression_frame = data.generate_cell_line_expression_frame(expression_file)
    ic50_series = data.generate_ic_50_series(ic_50_filename)
    trimmed_expression_frame, trimmed_ic50_series = data.generate_cell_line_intersection(expression_frame,ic50_series)
    assert len(trimmed_expression_frame.columns) == len(trimmed_ic50_series.index)

def test_bin_ic50_series():
    ic50_series = data.generate_ic_50_series(ic_50_filename)
    binned_ic50 = data.bin_ic_50_series(ic50_series)
    assert len(ic50_series) == len(binned_ic50)
    assert all(x in [0,1,2] for x in binned_ic50)
    assert math.fabs(sum(x for x in binned_ic50) - len(binned_ic50)) < 2

def test_trim_undetermined_cell_lines():
    expression_frame = data.generate_cell_line_expression_frame(expression_file)
    binned_ic50 = data.bin_ic_50_series(data.generate_ic_50_series(ic_50_filename))
    expression_frame,binned_ic50 = data.generate_cell_line_intersection(expression_frame,binned_ic50)
    trimmed_expression_frame = data.trim_undetermined_cell_lines(expression_frame,binned_ic50)
    assert len(trimmed_expression_frame.columns) == len([x for x in binned_ic50 if not x == 1])