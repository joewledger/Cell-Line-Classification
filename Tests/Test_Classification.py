import Src.Classification as classify
import Src.DataFormatter as dfm
import numpy as np

expression_file = "Data/CCLE_Data/sample1000.csv"
ic50_file = 'Data/IC_50_Data/CL_Sensitivity_Multiple_Drugs.csv'
patient_directory = "Data/TCGA_Data/9f2c84a7-c887-4cb5-b6e5-d38b00d678b1/Expression-Genes/UNC__AgilentG4502A_07_3/Level_3"

def test_neat_accuracy():
    model = classify.Scikit_Model("neat")
    accuracy = model.get_model_accuracy_filter_feature_size(expression_file,ic50_file,5,1,"SMAP")
    acc = [a for a in accuracy]
    print(acc)
