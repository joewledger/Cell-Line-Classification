import DataFormatter as dfm
import Cross_Validator as cv
import Feature_Selection as fs
from sklearn.feature_selection import RFE

from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn import tree
from abc import ABCMeta, abstractmethod

"""
Cell Line Classification Project using SVMs (Support Vector Machines) and Neural Networks
Working with Elena Svenson under the direction of Dr. Mehmet Koyuturk
We have IC50 values for a bunch of different cell lines for the drug we are testing (SMAPs -- Small Molecular Activators of PP2A)
We are going to apply SVM to classify cell lines as either sensitive or resistant to this drug
The training input values are the gene expression measurements for each cell line
The training output values are the IC50 values discretized into several bins: "sensitive", "undetermined" , and "resistant"
"""

class Generic_Scikit_Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def construct_model(self,**kwargs):
        return None

    @abstractmethod
    def get_model_accuracy_filter_threshold(self,expression_file, ic50_file,threshold,num_permutations,**kwargs):
        model = self.construct_model(**kwargs)
        scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=threshold)
        for i in range(0,num_permutations):
            shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
            yield cv.cross_val_score_filter_feature_selection(model,cv.trim_X_threshold,threshold,shuffled_data,shuffled_target,cv=5).mean()

    @abstractmethod
    def get_model_accuracy_filter_feature_size(self,expression_file, ic50_file,feature_size,num_permutations,**kwargs):
        model = self.construct_model(**kwargs)
        scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=None)
        for i in range(0,num_permutations):
            shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
            accuracy = cv.cross_val_score_filter_feature_selection(model,cv.trim_X_num_features,feature_size,shuffled_data,shuffled_target,cv=5)
            yield accuracy.mean()

    @abstractmethod
    def get_predictions_full_CCLE_dataset(self,expression_file,ic50_file,threshold,**kwargs):
        model = self.construct_model(**kwargs)
        training_frame,training_series = dfm.get_expression_frame_and_ic50_series(expression_file,ic50_file,normalized=True,trimmed=True,threshold=threshold)
        training_data,training_target = dfm.get_scikit_data_and_target(training_frame,training_series)

        cell_lines, testing_data = dfm.get_normalized_full_expression_identifiers_and_data(expression_file,training_frame.index)

        model.fit(training_data,training_target)
        predictions = model.predict(testing_data)

        return cell_lines, predictions

    @abstractmethod
    def get_predictions_full_CCLE_dataset_from_top_features(self,expression_file,ic50_file,num_features,**kwargs):
        model = self.construct_model(**kwargs)
        expression_frame,ic50_series = dfm.get_expression_frame_and_ic50_series(expression_file,ic50_file,normalized=True,trimmed=True)
        top_features = dfm.get_pval_top_n_features(expression_frame,ic50_series,num_features)
        expression_frame = expression_frame.ix[top_features]
        scikit_data,scikit_target = dfm.get_scikit_data_and_target(expression_frame,ic50_series)

        cell_lines, testing_data = dfm.get_normalized_full_expression_identifiers_and_data(expression_file,expression_frame.index)
        model.fit(scikit_data,scikit_target)
        predictions = model.predict(testing_data)

        return cell_lines,predictions,list(top_features)

    @abstractmethod
    def get_model_coefficients(self,expression_file,ic50_file,threshold,**kwargs):
        model = self.construct_model(**kwargs)
        scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=threshold)
        model.fit(scikit_data,scikit_target)
        return model.coef_[0]

    @abstractmethod
    def get_patient_predictions(self,expression_file,ic50_file,patient_directory,threshold,**kwargs):
        """
        Returns the predictions for which patients are likely to be sensitive to SMAPs and which are likely to be resistant.
        First trains a given SVM model on expression data, and then uses the trained model to predict patient outcome.

        Returns a list of patient identifiers, and a list of predictions about the patients response to a given drug.
        """
        trimmed = kwargs.pop('trimmed',False)
        model = self.construct_model(**kwargs)

        e_data,e_target,p_identifiers,p_data = dfm.get_cell_line_and_patient_expression_data_target(expression_file,ic50_file,patient_directory,threshold,trimmed=trimmed)

        model.fit(e_data,e_target)
        predictions = model.predict(p_data)

        return p_identifiers,predictions

    @abstractmethod
    def get_patient_predictions_top_features(self,expression_file,ic50_file,patient_directory,num_features,**kwargs):
        trimmed = kwargs.pop('trimmed',False)
        model = self.construct_model(**kwargs)

        e_data,e_target,p_identifiers,p_data,top_features = dfm.get_cell_line_and_patient_expression_data_target_top_features(expression_file,ic50_file,patient_directory,num_features,trimmed=trimmed)

        model.fit(e_data,e_target)
        predictions = model.predict(p_data)

        return p_identifiers,predictions,top_features




    @abstractmethod
    def get_model_accuracy_bidirectional_feature_search(self,expression_file,ic50_file,target_features,num_permutations,**kwargs):
        """
        Returns a generator which yields accuracy scores for bidirectional feature searches
        """
        expression_frame, ic50_series = dfm.get_expression_frame_and_ic50_series(expression_file, ic50_file,normalized=True,trimmed=True)

        for i in xrange(0,num_permutations):
            model = self.construct_model(**kwargs)

            features_selected = fs.bidirectional_feature_search(model,expression_frame,ic50_series,target_features)
            scikit_data,scikit_target = dfm.get_scikit_data_and_target(dfm.get_expression_frame_with_features(expression_frame,features_selected),ic50_series)
            shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
            model = self.construct_model(**kwargs)
            yield cross_val_score(model,shuffled_data,shuffled_target,cv=5).mean()

    @abstractmethod
    def get_model_accuracy_RFE(self,expression_file,ic50_file,target_features,num_permutations,**kwargs):
        scikit_data,scikit_target = dfm.get_expression_scikit_data_target(expression_file,ic50_file,normalized=True,trimmed=True,threshold=None)
        step_length = int(len(scikit_data.tolist()[0]) / 100) + 1
        for i in xrange(0,num_permutations):
            model = self.construct_model(**kwargs)
            shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
            selector = RFE(model,target_features,step=step_length)
            yield cross_val_score(selector,shuffled_data,shuffled_target,cv=5).mean()

class Decision_Tree_Model(Generic_Scikit_Model):

    def __init__(self):
        self.description = "A decision tree classification model."

    def construct_model(self,**kwargs):
        return tree.DecisionTreeClassifier(**kwargs)

    def get_model_accuracy_filter_threshold(self,expression_file, ic50_file,threshold,num_permutations,**kwargs):
        return super(Decision_Tree_Model, self).get_model_accuracy_filter_threshold(expression_file, ic50_file,threshold,num_permutations,**kwargs)

    def get_model_accuracy_filter_feature_size(self,expression_file, ic50_file,threshold,num_permutations,**kwargs):
        return super(Decision_Tree_Model,self).get_model_accuracy_filter_feature_size(expression_file, ic50_file,threshold,num_permutations,**kwargs)

    def get_predictions_full_CCLE_dataset(self,expression_file,ic50_file,threshold,**kwargs):
        return super(Decision_Tree_Model,self).get_predictions_full_CCLE_dataset(expression_file,ic50_file,threshold,**kwargs)

    def get_predictions_full_CCLE_dataset_from_top_features(self,expression_file,ic50_file,num_features,**kwargs):
        raise NotImplementedError

    def get_model_coefficients(self,expression_file,ic50_file,threshold,**kwargs):
        """
        Not implemented because the decision tree does not have model coefficients in the same way that the SVM
        linear model does.
        """
        raise NotImplementedError

    def get_patient_predictions(self,expression_file,ic50_file,patient_directory,threshold,**kwargs):
        return super(Decision_Tree_Model,self).get_patient_predictions(expression_file,ic50_file,patient_directory,threshold,**kwargs)

    def get_patient_predictions_top_features(self,expression_file,ic50_file,patient_directory,num_features,**kwargs):
        raise NotImplementedError

    def get_model_accuracy_bidirectional_feature_search(self,expression_file,ic50_file,target_features,num_permutations,**kwargs):
        raise NotImplementedError

    def get_model_accuracy_RFE(self,expression_file,ic50_file,target_features,num_permutations,**kwargs):
        return super(Decision_Tree_Model,self).get_model_accuracy_RFE(expression_file,ic50_file,target_features,num_permutations,**kwargs)

class SVM_Model(Generic_Scikit_Model):

    def __init__(self):
        self.description = "A SVM classification model."

    def construct_model(self,**kwargs):
        return svm.SVC(**kwargs)

    def get_model_accuracy_filter_threshold(self,expression_file, ic50_file,threshold,num_permutations,**kwargs):
        return super(SVM_Model, self).get_model_accuracy_filter_threshold(expression_file, ic50_file,threshold,num_permutations,**kwargs)

    def get_model_accuracy_filter_feature_size(self,expression_file, ic50_file,threshold,num_permutations,**kwargs):
        return super(SVM_Model,self).get_model_accuracy_filter_feature_size(expression_file, ic50_file,threshold,num_permutations,**kwargs)

    def get_predictions_full_CCLE_dataset(self,expression_file,ic50_file,threshold,**kwargs):
        return super(SVM_Model,self).get_predictions_full_CCLE_dataset(expression_file,ic50_file,threshold,**kwargs)

    def get_predictions_full_CCLE_dataset_from_top_features(self,expression_file,ic50_file,num_features,**kwargs):
        return super(SVM_Model,self).get_predictions_full_CCLE_dataset_from_top_features(expression_file,ic50_file,num_features,**kwargs)

    def get_model_coefficients(self,expression_file,ic50_file,threshold,**kwargs):
        if(kwargs['kernel'] == 'linear'):
            return super(SVM_Model,self).get_model_coefficients(expression_file,ic50_file,threshold,**kwargs)
        else:
            #The SVM will not have coefficients for non-linear models.
            raise NotImplementedError

    def get_patient_predictions(self,expression_file,ic50_file,patient_directory,threshold,**kwargs):
        return super(SVM_Model,self).get_patient_predictions(expression_file,ic50_file,patient_directory,threshold,**kwargs)

    def get_patient_predictions_top_features(self,expression_file,ic50_file,patient_directory,num_features,**kwargs):
        return super(SVM_Model,self).get_patient_predictions_top_features(expression_file,ic50_file,patient_directory,num_features,**kwargs)

    def get_model_accuracy_bidirectional_feature_search(self,expression_file,ic50_file,target_features,num_permutations,**kwargs):
        if(kwargs['kernel'] == 'linear'):
            return super(SVM_Model,self).get_model_accuracy_bidirectional_feature_search(expression_file,ic50_file,target_features,num_permutations,**kwargs)
        else:
            raise NotImplementedError

    def get_model_accuracy_RFE(self,expression_file,ic50_file,target_features,num_permutations,**kwargs):
        return super(SVM_Model,self).get_model_accuracy_RFE(expression_file,ic50_file,target_features,num_permutations,**kwargs)

class Neural_Network_Model(Generic_Scikit_Model):

    def __init__(self):
        self.description = "A Neural Network classification model."

    def construct_model(self,**kwargs):
        raise NotImplementedError

    def get_model_accuracy_filter_threshold(self,expression_file, ic50_file,threshold,num_permutations,**kwargs):
        raise NotImplementedError

    def get_model_accuracy_filter_feature_size(self,expression_file, ic50_file,threshold,num_permutations,**kwargs):
        raise NotImplementedError

    def get_predictions_full_CCLE_dataset_from_top_features(self,expression_file,ic50_file,num_features,**kwargs):
        raise NotImplementedError

    def get_predictions_full_CCLE_dataset(self):
        raise NotImplementedError

    def get_model_coefficients(self):
        raise NotImplementedError

    def get_patient_predictions(self):
        raise NotImplementedError

    def get_patient_predictions_top_features(self,expression_file,ic50_file,patient_directory,num_features,**kwargs):
        raise NotImplementedError

    def get_model_accuracy_bidirectional_feature_search(self,expression_file,ic50_file,target_features,num_permutations,**kwargs):
        raise NotImplementedError

    def get_model_accuracy_RFE(self,expression_file,ic50_file,target_features,num_permutations,**kwargs):
        raise NotImplementedError
