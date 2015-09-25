import Src.DataFormatter as dfm
import Src.Classification as classify
import Src.Cross_Validator as cv
import math

from sklearn.cross_validation import cross_val_score

expression_file = "Data/CCLE_Data/sample1000.res"
ic50_file = "Data/IC_50_Data/CL_Sensitivity.txt"

def test_cross_val_score_filter_feature_selection_threshold():

    threshold = 1.0
    scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file, ic50_file,normalized=True,trimmed=True,threshold=None)
    model = classify.construct_svc_model(kernel='linear')
    non_thresholded_test_1 = cv.cross_val_score_filter_feature_selection(model,cv.trim_X_threshold,threshold,scikit_data,scikit_target,cv=5)

    m = classify.construct_svc_model(kernel='linear')
    s_data,s_target = dfm.get_expression_scikit_data_target(expression_file, ic50_file,normalized=True,trimmed=True,threshold=threshold)
    non_thresholded_test_2 = cross_val_score(m,s_data,s_target,cv=5)

    threshold = .05
    scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file, ic50_file,normalized=True,trimmed=True,threshold=None)
    model = classify.construct_svc_model(kernel='linear')
    thresholded_test_1 = cv.cross_val_score_filter_feature_selection(model,cv.trim_X_threshold,threshold,scikit_data,scikit_target,cv=5)

    m = classify.construct_svc_model(kernel='linear')
    s_data,s_target = dfm.get_expression_scikit_data_target(expression_file, ic50_file,normalized=True,trimmed=True,threshold=threshold)
    thresholded_test_2 = cross_val_score(m,s_data,s_target,cv=5)


    #The non-thresholded tests should be the same because if we are not thresholding, it doesn't matter where we perform thresholding
    assert(math.fabs(non_thresholded_test_1.mean() - non_thresholded_test_2.mean()) < .001)

    #The first non_thresholded test should have lower accuracy because we are doing thresholding within the cross-validation,
    #which will reduce cross-validation overfitting and as a consequence reported cross-validation accuracy.
    assert(thresholded_test_1.mean() - thresholded_test_2.mean() < 0)


def test_cross_val_score_filter_feature_selection_feature_ranks():
    #Calculate cross-val score for feature selection down to 10 features.
    num_features = 10
    scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file, ic50_file,normalized=True,trimmed=True,threshold=None)
    model = classify.construct_svc_model(kernel='linear')
    cv_score_10_features = cv.cross_val_score_filter_feature_selection(model,cv.trim_X_num_features,num_features,scikit_data,scikit_target,cv=5)

    #Calculate cross-val score with no feature selection but using cv.cross_val_score_filter_feature_selection
    scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file, ic50_file,normalized=True,trimmed=True,threshold=None)
    num_features = len(scikit_data.tolist()[0])
    model = classify.construct_svc_model(kernel='linear')
    cv_score_all_features_1 = cv.cross_val_score_filter_feature_selection(model,cv.trim_X_num_features,num_features,scikit_data,scikit_target,cv=5)

    scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file, ic50_file,normalized=True,trimmed=True,threshold=None)
    model = classify.construct_svc_model(kernel='linear')
    cv_score_all_features_2 = cross_val_score(model,scikit_data,scikit_target,cv=5)

    #Feature selecction with only 10 features should have lower accuracy than feature selection with all features.
    assert(cv_score_10_features.mean() - cv_score_all_features_1.mean() < 0)

    #Feature selection with all features should have the same accuracy regardless of method of feature selection
    assert(math.fabs(cv_score_all_features_1.mean() - cv_score_all_features_2.mean()) < .001)