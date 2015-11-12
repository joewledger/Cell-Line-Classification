import DataFormatter as dfm
import Cross_Validator as cv
from sklearn.feature_selection import RFE

from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn import tree

"""
Cell Line Classification Project using SVMs (Support Vector Machines) and Neural Networks
Working with Elena Svenson under the direction of Dr. Mehmet Koyuturk
We have IC50 values for a bunch of different cell lines for the drug we are testing (SMAPs -- Small Molecular Activators of PP2A)
We are going to apply SVM to classify cell lines as either sensitive or resistant to this drug
The training input values are the gene expression measurements for each cell line
The training output values are the IC50 values discretized into several bins: "sensitive", "undetermined" , and "resistant"
"""

class Scikit_Model():

    def __init__(self,model_type,**kwargs):
        self.allowed_model_types = ['svm','dt','nn', 'neat']
        self.model_type = model_type
        self.kernel = (kwargs['kernel'] if 'kernel' in kwargs else None)
        if(not model_type in self.allowed_model_types):
            raise ValueError("Incorrect model type provided")
        if(model_type == 'svm'):
            self.model = svm.SVC(**kwargs)
        elif(model_type == 'dt'):
            self.model = tree.DecisionTreeClassifier(**kwargs)
        elif(model_type == 'nn'):
            raise NotImplementedError("Neural networks not yet implemented")
        elif(model_type == 'neat'):
            raise NotImplementedError("NEAT not yet implemented")

    def get_model_accuracy_filter_threshold(self,expression_file, ic50_file,threshold,num_permutations,drug):
        scikit_data,scikit_target = dfm.get_expression_scikit_data_target_for_drug(expression_file,ic50_file,drug,normalized=True,trimmed=True,threshold=threshold)
        for i in range(0,num_permutations):
            shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
            yield cv.cross_val_score_filter_feature_selection(self.model,cv.trim_X_threshold,threshold,shuffled_data,shuffled_target,cv=5).mean()

    def get_model_accuracy_filter_feature_size(self,expression_file, ic50_file,feature_size,num_permutations,drug):
        scikit_data,scikit_target = dfm.get_expression_scikit_data_target_for_drug(expression_file,ic50_file,drug,normalized=True,trimmed=True,threshold=None)
        for i in range(0,num_permutations):
            shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
            accuracy = cv.cross_val_score_filter_feature_selection(self.model,cv.trim_X_num_features,feature_size,shuffled_data,shuffled_target,cv=5)
            yield accuracy.mean()

    def get_model_accuracy_RFE(self,expression_file,ic50_file,target_features,num_permutations,drug):

        scikit_data,scikit_target = dfm.get_expression_scikit_data_target_for_drug(expression_file,ic50_file,drug,normalized=True,trimmed=True,threshold=None)
        step_length = int(len(scikit_data.tolist()[0]) / 100) + 1
        for i in xrange(0,num_permutations):
            shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
            selector = RFE(self.model,target_features,step=step_length)
            yield cross_val_score(selector,shuffled_data,shuffled_target,cv=5).mean()

    def get_model_coefficients_threshold(self,expression_file,ic50_file,threshold,drug):
        if(self.model_type == 'svm' and self.kernel == 'linear'):
            expression_frame,ic50_series = dfm.get_expression_frame_and_ic50_series_for_drug(expression_file, ic50_file,drug,normalized=True,trimmed=True,threshold=threshold)
            scikit_data,scikit_target = dfm.get_scikit_data_and_target(expression_frame,ic50_series)
            self.model.fit(scikit_data,scikit_target)
            return expression_frame.index, self.model.coef_[0]
        else:
            raise Exception("Method only defined for the SVM linear model")

    def get_model_RFE_top_features(self,expression_file,ic50_file,target_features,drug):
        expression_frame,ic50_series = dfm.get_expression_frame_and_ic50_series_for_drug(expression_file, ic50_file,drug,normalized=True,trimmed=True,threshold=None)
        scikit_data,scikit_target = dfm.get_scikit_data_and_target(expression_frame,ic50_series)
        step_length = int(len(scikit_data.tolist()[0]) / 100) + 1
        selector = RFE(self.model,target_features,step=step_length)
        selector.fit(scikit_data,scikit_target)
        return expression_frame.index, selector.support_

    def get_predictions_full_CCLE_dataset_threshold(self,expression_file,ic50_file,threshold,drug):
        training_frame,training_series = dfm.get_expression_frame_and_ic50_series_for_drug(expression_file,ic50_file,drug,normalized=True,trimmed=True,threshold=threshold)
        training_data,training_target = dfm.get_scikit_data_and_target(training_frame,training_series)

        cell_lines, testing_data = dfm.get_normalized_full_expression_identifiers_and_data(expression_file,training_frame.index)

        self.model.fit(training_data,training_target)
        predictions = self.model.predict(testing_data)

        return cell_lines, predictions

    def get_predictions_full_CCLE_dataset_top_features(self,expression_file,ic50_file,num_features,drug):
        expression_frame,ic50_series = dfm.get_expression_frame_and_ic50_series_for_drug(expression_file,ic50_file,drug,normalized=True,trimmed=True)
        top_features = dfm.get_pval_top_n_features(expression_frame,ic50_series,num_features)
        expression_frame = expression_frame.ix[top_features]
        scikit_data,scikit_target = dfm.get_scikit_data_and_target(expression_frame,ic50_series)

        cell_lines, testing_data = dfm.get_normalized_full_expression_identifiers_and_data(expression_file,expression_frame.index)
        self.model.fit(scikit_data,scikit_target)
        predictions = self.model.predict(testing_data)

        return cell_lines,predictions,list(top_features)

    def get_patient_predictions_threshold(self,expression_file,ic50_file,patient_directory,threshold,drug):
        """
        Returns the predictions for which patients are likely to be sensitive to SMAPs and which are likely to be resistant.
        First trains a given SVM model on expression data, and then uses the trained model to predict patient outcome.

        Returns a list of patient identifiers, and a list of predictions about the patients response to a given drug.
        """
        e_data,e_target,p_identifiers,p_data = dfm.get_cell_line_and_patient_expression_data_target_for_drug(expression_file,ic50_file,patient_directory,threshold,drug)

        self.model.fit(e_data,e_target)
        predictions = self.model.predict(p_data)

        return p_identifiers,predictions

    def get_patient_predictions_top_features(self,expression_file,ic50_file,patient_directory,num_features,drug):

        e_data,e_target,p_identifiers,p_data,top_features = dfm.get_cell_line_and_patient_expression_data_target_top_features_for_drug(expression_file,ic50_file,patient_directory,num_features,drug)

        self.model.fit(e_data,e_target)
        predictions = self.model.predict(p_data)

        return p_identifiers,predictions,top_features