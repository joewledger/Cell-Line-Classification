import argparse
from argparse import RawTextHelpFormatter
import datetime
import os
import DataFormatter as dfm
import Classification as classify
import Plotting as plt

"""
Writes results to a results directory

"""

def main():
    description = """
    A tool for predicting patient response to cancer drugs using several different machine learning models.\n
    These include Support Vector Machines, Artificial Neural Networks, and the NEAT algorithm.\n
    Depending on the parameters given, this module will run different experiments.\n
    \t0 : Graph SVM Linear kernel accuracy vs. threshold\n
    \t1 : Graph SVM RBF kernel accuracy vs. threshold\n
    \t2 : Graph SVM Polynomial kernel accuracy vs. threshold\n
    \t3 : Graph all SVM kernel accuracies vs. threshold on same graph\n
    \t4 : Save SVM Linear kernel model coefficients
    """
    parser = argparse.ArgumentParser(description=description,formatter_class=RawTextHelpFormatter)
    parser.add_argument('--experiments',nargs='+',type=int,help='The experiments to run.')
    parser.set_defaults(experiments=[x for x in xrange(0,5)])
    args = parser.parse_args()
    
    base_results_directory,expression_filename,ic50_filename,thresholds = define_parameters()
    results_directory = get_results_filepath(base_results_directory)
    make_results_directory_and_subdirectories(base_results_directory,results_directory)

    linear_acc = save_svm_accuracy_threshold_graph(results_directory, expression_filename,ic50_filename,thresholds,model_parameters={'kernel' : 'linear'}) if (0 in args.experiments) else None
    rbf_acc = save_svm_accuracy_threshold_graph(results_directory, expression_filename,ic50_filename,thresholds,model_parameters={'kernel' : 'rbf'}) if (1 in args.experiments) else None
    poly_acc = save_svm_accuracy_threshold_graph(results_directory, expression_filename,ic50_filename,thresholds,model_parameters={'kernel' : 'poly'}) if (2 in args.experiments) else None
    if (3 in args.experiments):
        save_svm_accuracy_threshold_graph_multiple_kernels(results_directory,linear_acc,rbf_acc,poly_acc)
    if (4 in args.experiments):
        save_svm_model_coefficients(results_directory,expression_filename,ic50_filename,thresholds)



def define_parameters():
    base_results_directory = os.path.dirname(__file__) + '/../Results/'
    expression_filename = os.path.dirname(__file__) + '/../Data/CCLE_Data/sample1000.res'
    ic50_filename = os.path.dirname(__file__) + '/../Data/IC_50_Data/CL_Sensitivity.txt'
    thresholds = [float(x) * .01 for x in xrange(1,5)]
    return base_results_directory,expression_filename,ic50_filename,thresholds

def get_results_filepath(base_results_directory):
    """
    Gets the filepath for a new results directory based on the current date and time
    """
    curr_time = str(datetime.datetime.today()).replace(" ","_").replace(":","-")
    curr_time = curr_time[:curr_time.rfind("-")]
    return base_results_directory + curr_time + "/"

def make_results_directory_and_subdirectories(base_results_directory,results_directory):
    if not os.path.isdir(base_results_directory):
        os.mkdir(base_results_directory)
    os.mkdir(results_directory)
    os.mkdir(results_directory + "Plots")
    os.mkdir(results_directory + "Plots/SVM_Accuracies")
    os.mkdir(results_directory + "Model_Coefficients")

def save_svm_accuracy_threshold_graph(results_directory,expression_file,ic50_file,thresholds,model_parameters={'kernel' : 'linear'}):
    model = classify.construct_svc_model(**model_parameters)
    accuracies = classify.get_svm_model_accuracy_multiple_thresholds(model, expression_file, ic50_file, thresholds)
    outfile = results_directory + "Plots/SVM_Accuracies/%s_accuracy_threshold.png" % str(model.kernel)
    plt.plot_accuracy_threshold_curve(outfile,[x[0] for x in accuracies],[x[1] for x in accuracies])
    return accuracies

def save_svm_accuracy_threshold_graph_multiple_kernels(results_directory, linear,rbf,poly):
    outfile = results_directory + "Plots/SVM_Accuracies/multiple_kernel_accuracy_threshold.png"
    kernels = [linear,rbf,poly]
    plt.plot_accuracy_threshold_multiple_kernels(outfile,kernels)

def save_svm_model_coefficients(results_directory, expression_file,ic50_file,thresholds):
    results_file = results_directory + "Model_Coefficients/svm_linear.txt"
    writer = open(results_file,"wb")
    for threshold in thresholds:
        writer.write("Threshold: %s\n" % str(threshold))
        model = classify.construct_svc_model(kernel="linear")
        expression_frame,ic50_series = dfm.generate_trimmed_thresholded_normalized_expression_frame(expression_file,ic50_file,threshold)
        genes = list(expression_frame.index)
        model_coefficients = classify.get_svm_model_coefficients(model,expression_file,ic50_file,threshold)
        writer.write("\t".join(str(gene) for gene in genes) + "\n")
        writer.write("\t".join(str(coef) for coef in model_coefficients)  + "\n\n")
    writer.close()

if __name__ == '__main__':
    main()










