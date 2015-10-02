import Src.Classification as classify
import Src.DataFormatter as dfm
import Src.Feature_Selection as fs

expression_file = "Data/CCLE_Data/sample100.res"
ic50_file = "Data/IC_50_Data/CL_Sensitivity.txt"


def test_bidirectional_feature_selection():
    model_object = classify.SVM_Model()
    model = model_object.construct_model(kernel='linear')
    expression_frame,ic50_series = dfm.get_expression_frame_and_ic50_series(expression_file, ic50_file,normalized=True,trimmed=True)
    print(fs.bidirectional_feature_search(model,expression_frame,ic50_series,10))

def test_backward_step():
    model_object = classify.SVM_Model()
    model = model_object.construct_model(kernel='linear')
    expression_frame,ic50_series = dfm.get_expression_frame_and_ic50_series(expression_file, ic50_file,normalized=True,trimmed=True)
    backward_step_size = 10
    forward_features_selected = []
    backward_features_removed = []

    assert len(fs.backward_step(model,expression_frame,ic50_series,backward_step_size,forward_features_selected,backward_features_removed)) == 10

def test_forward_step():
    model_object = classify.SVM_Model()
    model = model_object.construct_model(kernel='linear')
    expression_frame,ic50_series = dfm.get_expression_frame_and_ic50_series(expression_file, ic50_file,normalized=True,trimmed=True)
    forward_features_selected = []
    backward_features_removed = []

    print(fs.forward_step(model,expression_frame,ic50_series,forward_features_selected,backward_features_removed))