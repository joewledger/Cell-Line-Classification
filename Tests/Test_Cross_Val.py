import Src.DataFormatter as dfm
import Src.Classification as classify
import Src.Cross_Validator as cv

from sklearn.cross_validation import cross_val_score

expression_file = "Data/CCLE_Data/sample1000.res"
ic50_file = "Data/IC_50_Data/CL_Sensitivity.txt"

def test_cross_val_score_filter_feature_selcetion():
    threshold = 1.0
    scikit_data,scikit_target = dfm.generate_trimmed_normalized_scikit_data_and_target(expression_file,ic50_file)
    model = classify.construct_svc_model(kernel='linear')
    print(cv.cross_val_score_filter_feature_selection(model,threshold,scikit_data,scikit_target,cv=5).mean())

    m = classify.construct_svc_model(kernel='linear')
    s_data,s_target = dfm.generate_trimmed_thresholded_normalized_scikit_data_and_target(expression_file,ic50_file,threshold)
    print(cross_val_score(m,s_data,s_target,cv=5).mean())