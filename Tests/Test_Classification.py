import Src.Classification as classify
from sklearn import svm

expression_file = "Data/CCLE_Data/sample1000.res"
ic50_filename = "Data/IC_50_Data/CL_Sensitivity.txt"

def test_get_svm_model_accuracy():
    model = svm.SVC(kernel='linear')
    scores = classify.get_svm_model_accuracy(model,expression_file,ic50_filename,.05)
    assert len(scores) == 5
    assert all(0.0 <= x <= 1.0 for x in scores)
    pass

def test_get_svm_model_accuracy_multiple_thresholds():
    model = svm.SVC(kernel='linear')
    scores = classify.get_svm_model_accuracy_multiple_thresholds(model,expression_file,ic50_filename,[float(x) * .05 for x in xrange(1,21)])
    assert all(0.0 <= x[0] <= 1.0 for x in scores)
    assert all(len(x[1]) == 2 for x in scores)
    assert all(0.0 <= x[1][0] <= 1.0 for x in scores)
    assert all(0.0 <= x[1][1] <= 1.0 for x in scores)
    pass