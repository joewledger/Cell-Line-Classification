import Src.DataFormatter as dfm
import Src.Classification as classify
import Src.Cross_Validator as cv

expression_file = "Data/CCLE_Data/sample1000.res"
ic50_file = "Data/IC_50_Data/CL_Sensitivity.txt"

def test_cross_val():
    threshold = .05
    scikit_data,scikit_target = dfm.generate_trimmed_thresholded_normalized_scikit_data_and_target(expression_file,ic50_file,threshold)
    model = classify.construct_svc_model(kernel='linear')
    return cv.cross_val_score(model,scikit_data,scikit_target,cv=5).mean()