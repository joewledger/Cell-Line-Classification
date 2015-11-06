import Src.DataFormatter as data
import os
import math
import numpy as np

patient_directory = "Data/TCGA_Data/9f2c84a7-c887-4cb5-b6e5-d38b00d678b1/Expression-Genes/UNC__AgilentG4502A_07_3/Level_3"
#expression_file = "Data/CCLE_Data/CCLE_Expression_2012-09-29.res"
expression_file = "Data/CCLE_Data/sample1000.res"
ic50_file = "Data/IC_50_Data/CL_Sensitivity.txt"

def test_get_ic50_series_for_drug():
    print(data.get_ic50_series_for_drug("Data/IC_50_Data/CL_Sensitivity_Multiple_Drugs.csv","Erlotinib"))

def test_get_expression_frame_and_ic50_series_for_drug():
    print(data.get_expression_frame_and_ic50_series_for_drug(expression_file, "Data/IC_50_Data/CL_Sensitivity_Multiple_Drugs.csv","Erlotinib",normalized=False,trimmed=True,threshold=None))