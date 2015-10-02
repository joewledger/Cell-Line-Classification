import sklearn.cross_validation as cv
import DataFormatter as dfm
import math
import copy
"""
A python module for doing more advanced feature selection techniques such as bidirectional search.
"""

def bidirectional_feature_search(model,expression_frame,ic50_series,target_features):
    #Determine forward and backward step sizes based on how large the steps will need to be to
    num_features = len(expression_frame.index)
    forward_step_size,backward_step_size = calculate_forward_backward_step_sizes(num_features,target_features)
    forward_features_selected, backward_features_removed = [],[]

    for step in xrange(0,calculate_num_steps(forward_step_size,target_features)):
        forward_features_selected = forward_step(model,expression_frame,ic50_series,forward_features_selected,backward_features_removed)
        backward_features_removed = backward_step(model,expression_frame,ic50_series,backward_step_size,forward_features_selected,backward_features_removed)

    return set(forward_features_selected).intersection(set(expression_frame.index) - set(backward_features_removed))

def forward_step(model,expression_frame,ic50_series,forward_features_selected,backward_features_removed):
    potential_features = set(expression_frame.index) - set(forward_features_selected) - set(backward_features_removed)
    max_score = -1
    best_feature = None
    for feature in potential_features:
        model = copy.copy(model)
        model_features = set(forward_features_selected) & set(feature)
        expression_frame = dfm.get_expression_frame_with_features(expression_frame,model_features)
        scikit_data,scikit_target = dfm.get_scikit_data_and_target(expression_frame,ic50_series)
        score = cv.cross_val_score(model,scikit_data,scikit_target,cv=5).mean()
        if(score > max_score):
            max_score = score
            best_feature = feature
    if(best_feature):
        forward_features_selected.append(best_feature)
        return forward_features_selected
    else:
        return forward_features_selected

def backward_step(model,expression_frame,ic50_series,backward_step_size,forward_features_selected,backward_features_removed):
    removable_features = set(expression_frame.index) - set(forward_features_selected) - set(backward_features_removed)
    expression_frame = dfm.get_expression_frame_with_features(expression_frame,set(expression_frame.index) - set(backward_features_removed))
    scikit_data,scikit_target = dfm.get_scikit_data_and_target(expression_frame,ic50_series)
    model.fit(scikit_data,scikit_target)
    coefs = model.coef_[0]
    feature_names = list(expression_frame.index)
    coefs,feature_names = zip(*sorted(zip(coefs,feature_names),key=lambda x : math.fabs(x[0])))
    num_features_removed = 0
    feature_index = 0
    while(num_features_removed < backward_step_size and feature_index < len(feature_names)):
        if(feature_names[feature_index] in removable_features):
            backward_features_removed.append(feature_names[feature_index])
            num_features_removed += 1
        feature_index += 1
    return backward_features_removed

def calculate_forward_backward_step_sizes(num_features,target_features):
    forward_step_size = 1
    backward_step_size = (num_features - target_features) / target_features + 1
    return forward_step_size, backward_step_size


def calculate_num_steps(forward_step_size,target_features):
    return int(target_features / forward_step_size)