#!/usr/bin/env python2
"""Demo for Experiment class."""

from experiment import Experiment
import SVM_Classification as svmc

class MultiProcessing(Experiment):

    def __init__(self, kernel):
        super().__init__()
        self._params['Trial'] = list(range(1000))
        self.kernel = kernel

    def task(self,configuration):
        trial = configuration[0]
        ic50_file = "IC_50_Data/CL_Sensitivity.txt"
        #expression_features_filename = "CCLE_Data/CCLE_Expression_2012-09-29.res"
        expression_file = "CCLE_Data/sample1000.res"
        svm = svmc.SVM_Classification(ic50_file,expression_file,exclude_undetermined=True,kernel=self.kernel,thresholds=[float(i) * .01 for i in range(1,21)])
        all_predictions,all_features, all_evaluations = svm.evaluate_all_thresholds(5)
        accuracy_values = [svm.model_accuracy(evaluation) for evaluation in all_evaluations]
        accuracy_values_sensitive = [svm.model_accuracy_sensitive(evaluation) for evaluation in all_evaluations]
        return kernel,thresholds, accuracy_values, accuracy_values_sensitive


    def result(self,retval):
        with open("Accuracy_Permutation.txt","a") as infile:
            infile.write(str(retval) + "\n")
        self.results.append(retval)

if __name__=='__main__':
    exp = MultiProcessing('linear')
    exp.run(nproc=1)
    exp = MultiProcessing('poly')
    exp.run(nproc=1)
    exp = MultiProcessing('linear')
    exp.run(nproc=1)