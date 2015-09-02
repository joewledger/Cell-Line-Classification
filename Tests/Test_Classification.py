import Src.Classification as classify

expression_file = "Data/CCLE_Data/sample1000.res"
ic50_filename = "Data/IC_50_Data/CL_Sensitivity.txt"

def test_get_svm_model_accuracy():
    print(classify.get_svm_model_accuracy(expression_file,ic50_filename))