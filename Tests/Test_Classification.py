import Src.Classification as classify
import Src.DataFormatter as dfm

expression_file = "Data/CCLE_Data/sample1000.res"
ic50_file = "Data/IC_50_Data/CL_Sensitivity.txt"

def test_get_svm_model_accuracy():
    model = classify.construct_svc_model(kernel='linear')
    scores = classify.get_svm_model_accuracy(model,expression_file,ic50_file,.05,5)
    assert len(scores) == 5
    assert all(0.0 <= x <= 1.0 for x in scores)
    assert any(not x == scores[0] for x in scores)
    pass

def test_get_svm_model_accuracy_multiple_thresholds():
    num_thresholds = 5
    num_permutations = 5
    thresholds = [float(x) * .05 for x in xrange(1,num_thresholds + 1)]

    model = classify.construct_svc_model(kernel='linear')
    scores = classify.get_svm_model_accuracy_multiple_thresholds(model,expression_file,ic50_file,thresholds,num_permutations)

    assert all(all(0.0 <= x <= 1.0 for x in array) for array in scores.values())
    assert len(scores.keys()) == num_thresholds
    assert all(len(value) == num_permutations for value in scores.values())
    pass

def test_get_svm_model_coefficients():
    threshold = .05
    model = classify.construct_svc_model(kernel='linear')
    parameters = classify.get_svm_model_coefficients(model,expression_file,ic50_file,threshold)
    expression_frame,ic50_series = dfm.generate_trimmed_thresholded_normalized_expression_frame(expression_file,ic50_file,threshold)
    assert len(parameters) == len(expression_frame.index)
    pass

def test_get_patient_predictions():
    model = classify.construct_svc_model(kernel='linear')
    predictions = classify.get_svm_patient_predictions()
    assert False