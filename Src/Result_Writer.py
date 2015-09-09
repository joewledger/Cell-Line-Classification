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
    parser.add_argument('--results_dir',type=str,help='The directory to save results to.')
    parser.add_argument('--expression_file',type=str,help='The file to read gene expression measurements from (Use \'full\' for full dataset).')
    parser.add_argument('--ic50_file',type=str,help='The file to read the IC50 measurements from.')
    parser.add_argument('--patient_dir',type=str,help='The directory containing the patient information')
    parser.add_argument('--threshold_increment',type=float,help='The increment to test thresholds at.')
    parser.add_argument('--num_thresholds',type=int,help='The number of thresholds to test.')
    parser.add_argument('--num_permutations',type=int,help='The number of permutations to use for cross-fold validation.')
    parser.set_defaults(**default_parameters())
    args = parser.parse_args()

    results_directory = get_results_filepath(args.results_dir)
    make_results_directory_and_subdirectories(args.results_dir,results_directory)
    expression_file = args.full_expression_file if args.expression_file == "full" else args.expression_file

    thresholds = [args.threshold_increment * x for x in xrange(1,args.num_thresholds + 1)]
    run_experiments(results_directory,args.experiments,expression_file,args.ic50_file,args.patient_dir,thresholds,args.num_permutations)


def run_experiments(results_directory, experiments, expression_filename,ic50_filename,patient_directory, thresholds,num_permutations):
    linear_acc = save_svm_accuracy_threshold_graph(results_directory, expression_filename,ic50_filename,thresholds,num_permutations,model_parameters={'kernel' : 'linear'}) if (0 in experiments) else None
    rbf_acc = save_svm_accuracy_threshold_graph(results_directory, expression_filename,ic50_filename,thresholds,num_permutations,model_parameters={'kernel' : 'rbf'}) if (1 in experiments) else None
    poly_acc = save_svm_accuracy_threshold_graph(results_directory, expression_filename,ic50_filename,thresholds,num_permutations,model_parameters={'kernel' : 'poly'}) if (2 in experiments) else None
    if (3 in experiments):
        save_svm_accuracy_threshold_graph_multiple_kernels(results_directory,linear_acc,rbf_acc,poly_acc)
    if (4 in experiments):
        save_svm_model_coefficients(results_directory,expression_filename,ic50_filename,thresholds)

def default_parameters():
    parameters = {}
    parameters['experiments'] = [x for x in xrange(0,5)]
    parameters['results_dir'] = os.path.dirname(__file__) + '/../Results/'
    parameters['expression_file'] = os.path.dirname(__file__) + '/../Data/CCLE_Data/sample1000.res'
    parameters['full_expression_file'] = os.path.dirname(__file__) + '/../Data/CCLE_Data/CCLE_Expression_2012-09-29.res'
    parameters['ic50_file'] = os.path.dirname(__file__) + '/../Data/IC_50_Data/CL_Sensitivity.txt'
    parameters['patient_dir'] = os.path.dirname(__file__) + "Data/TCGA_Data/9f2c84a7-c887-4cb5-b6e5-d38b00d678b1/Expression-Genes/UNC__AgilentG4502A_07_3/Level_3"
    parameters['threshold_increment'] = .01
    parameters['num_thresholds'] = 100
    parameters['num_permutations'] = 20
    return parameters

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

def save_svm_accuracy_threshold_graph(results_directory,expression_file,ic50_file,thresholds,num_permutations,model_parameters={'kernel' : 'linear'}):
    model = classify.construct_svc_model(**model_parameters)
    accuracies = classify.get_svm_model_accuracy_multiple_thresholds(model, expression_file, ic50_file, thresholds,num_permutations)
    outfile = results_directory + "Plots/SVM_Accuracies/%s_accuracy_threshold.png" % str(model.kernel)
    plt.plot_accuracy_threshold_curve(outfile,thresholds,accuracies,"SVM %s Kernel" % model_parameters['kernel'])
    return accuracies

def save_svm_accuracy_threshold_graph_multiple_kernels(results_directory, linear,rbf,poly):
    """
    Saves accuracy threshold graphs for multiple SVM kernels on the same set of axes.
    The parameters linear, rbf, and poly are dictionaries mapping threshold to a list of accuracy values for that particular kernels
    """
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