import Src.Classification as classify

expression_file = "Data/CCLE_Data/sample1000.res"
ic50_filename = "Data/IC_50_Data/CL_Sensitivity.txt"

def test_get_svm_model_accuracy():
    print(classify.get_svm_model_accuracy(expression_file,ic50_filename,.05))

def test_get_svm_model_accuracy_multiple_thresholds():
    print(classify.get_svm_model_accuracy_multiple_thresholds(expression_file,ic50_filename,[float(x) * .05 for x in xrange(1,21)]))