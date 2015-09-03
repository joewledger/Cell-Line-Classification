import Src.Classification as classify
import Src.DataFormatter as dfm

expression_file = "Data/CCLE_Data/sample1000.res"
ic50_file = "Data/IC_50_Data/CL_Sensitivity.txt"

def test_get_svm_model_accuracy():
    model = classify.construct_svc_model(kernel='linear')
    scores = classify.get_svm_model_accuracy(model,expression_file,ic50_file,.05)
    assert len(scores) == 5
    assert all(0.0 <= x <= 1.0 for x in scores)
    pass

def test_get_svm_model_accuracy_multiple_thresholds():
    model = classify.construct_svc_model(kernel='linear')
    scores = classify.get_svm_model_accuracy_multiple_thresholds(model,expression_file,ic50_file,[float(x) * .05 for x in xrange(1,21)])
    assert all(0.0 <= x[0] <= 1.0 for x in scores)
    assert all(len(x[1]) == 2 for x in scores)
    assert all(0.0 <= x[1][0] <= 1.0 for x in scores)
    assert all(0.0 <= x[1][1] <= 1.0 for x in scores)
    pass

def test_get_svm_model_coefficients():
    threshold = .05
    model = classify.construct_svc_model(kernel='linear')
    parameters = classify.get_svm_model_coefficients(model,expression_file,ic50_file,threshold)
    expression_frame,ic50_series = dfm.generate_trimmed_thresholded_normalized_expression_frame(expression_file,ic50_file,threshold)
    assert len(parameters) == len(expression_frame.index)
    pass