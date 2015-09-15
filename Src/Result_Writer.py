import argparse
from argparse import RawDescriptionHelpFormatter
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
    A tool for predicting patient response to cancer drugs using several different machine learning models.\n\n
    These include Support Vector Machines, Artificial Neural Networks, and the NEAT algorithm.\n\n
    Depending on the parameters given, this module will run different experiments.\n\n""" + get_experiment_descriptions()
    description = description.replace("    ","")

    parser = argparse.ArgumentParser(description=description,formatter_class=RawDescriptionHelpFormatter)
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
    log_file = results_directory + "log.txt"
    linear_acc,rbf_acc,poly_acc = None,None,None
    """
    if(0 in experiments):
        log(log_file,"Starting Experiment 0 at %s\n" % str(datetime.datetime.today()))
        linear_acc = save_svm_accuracy_threshold_graph(results_directory, expression_filename,ic50_filename,thresholds,num_permutations,model_parameters={'kernel' : 'linear'})
        log(log_file,"Finished Experiment 0 at %s\n" % str(datetime.datetime.today()))
    if(1 in experiments):
        log(log_file,"Starting Experiment 1 at %s\n" % str(datetime.datetime.today()))
        rbf_acc = save_svm_accuracy_threshold_graph(results_directory, expression_filename,ic50_filename,thresholds,num_permutations,model_parameters={'kernel' : 'rbf'})
        log(log_file,"Finished Experiment 1 at %s\n" % str(datetime.datetime.today()))
    if(2 in experiments):
        log(log_file,"Starting Experiment 2 at %s\n" % str(datetime.datetime.today()))
        poly_acc = save_svm_accuracy_threshold_graph(results_directory, expression_filename,ic50_filename,thresholds,num_permutations,model_parameters={'kernel' : 'poly'})
        log(log_file,"Finished Experiment 2 at %s\n" % str(datetime.datetime.today()))
    if (3 in experiments):
        log(log_file,"Starting Experiment 3 at %s\n" % str(datetime.datetime.today()))
        save_svm_accuracy_threshold_graph_multiple_kernels(results_directory,linear_acc,rbf_acc,poly_acc)
        log(log_file,"Finished Experiment 3 at %s\n" % str(datetime.datetime.today()))
    if (4 in experiments):
        log(log_file,"Starting Experiment 4 at %s\n" % str(datetime.datetime.today()))
        save_svm_model_coefficients(results_directory,expression_filename,ic50_filename,thresholds)
        log(log_file,"Finished Experiment 4 at %s\n" % str(datetime.datetime.today()))
    if 5 in experiments:
        log(log_file,"Starting Experiment 5 at %s\n" % str(datetime.datetime.today()))
        save_svm_patient_predictions(results_directory,expression_filename, ic50_filename,patient_directory, thresholds,model_parameters={'kernel' : 'linear'})
        log(log_file,"Finished Experiment 5 at %s\n" % str(datetime.datetime.today()))
    if 6 in experiments:
        log(log_file,"Starting Experiment 6 at %s\n" % str(datetime.datetime.today()))
        save_svm_patient_predictions(results_directory,expression_filename, ic50_filename,patient_directory, thresholds,model_parameters={'kernel' : 'rbf'})
        log(log_file,"Finished Experiment 6 at %s\n" % str(datetime.datetime.today()))
    if 7 in experiments:
        log(log_file,"Starting Experiment 7 at %s\n" % str(datetime.datetime.today()))
        save_svm_patient_predictions(results_directory,expression_filename, ic50_filename,patient_directory, thresholds,model_parameters={'kernel' : 'poly'})
        log(log_file,"Finished Experiment 7 at %s\n" % str(datetime.datetime.today()))
    if 8 in experiments:
        log(log_file,"Starting Experiment 8 at %s\n" % str(datetime.datetime.today()))
        save_svm_patient_predictions(results_directory,expression_filename, ic50_filename,patient_directory, thresholds,model_parameters={'kernel' : 'linear'},trimmed=True)
        log(log_file,"Finished Experiment 8 at %s\n" % str(datetime.datetime.today()))
    if 9 in experiments:
        log(log_file,"Starting Experiment 9 at %s\n" % str(datetime.datetime.today()))
        save_svm_patient_predictions(results_directory,expression_filename, ic50_filename,patient_directory, thresholds,model_parameters={'kernel' : 'rbf'},trimmed=True)
        log(log_file,"Finished Experiment 9 at %s\n" % str(datetime.datetime.today()))
    if 10 in experiments:
        log(log_file,"Starting Experiment 10 at %s\n" % str(datetime.datetime.today()))
        save_svm_patient_predictions(results_directory,expression_filename, ic50_filename,patient_directory, thresholds,model_parameters={'kernel' : 'poly'},trimmed=True)
        log(log_file,"Finished Experiment 10 at %s\n" % str(datetime.datetime.today()))
    if 11 in experiments:
        log(log_file,"Starting Experiment 11 at %s\n" % str(datetime.datetime.today()))
        save_svm_full_CCLE_dataset_predictions(results_directory,expression_filename,ic50_filename,thresholds,model_parameters={'kernel' : 'linear'})
        log(log_file,"Finished Experiment 11 at %s\n" % str(datetime.datetime.today()))
    if 12 in experiments:
        log(log_file,"Starting Experiment 12 at %s\n" % str(datetime.datetime.today()))
        save_svm_full_CCLE_dataset_predictions(results_directory,expression_filename,ic50_filename,thresholds,model_parameters={'kernel' : 'rbf'})
        log(log_file,"Finished Experiment 12 at %s\n" % str(datetime.datetime.today()))
    if 13 in experiments:
        log(log_file,"Starting Experiment 13 at %s\n" % str(datetime.datetime.today()))
        save_svm_full_CCLE_dataset_predictions(results_directory,expression_filename,ic50_filename,thresholds,model_parameters={'kernel' : 'poly'})
        log(log_file,"Finished Experiment 13 at %s\n" % str(datetime.datetime.today()))
    """

def define_experiments():
    """
    Returns a dictionary of tuples, where key is the experiment number, and each value is a tuple containing:
    1) A description of the experiment
    2) The unbound method corresponding to the experiment
    3) Any positional args to be passed to the unbound method
    4) Any keyword args to be passed to the unbound method (optional)
    5) The variable name to save the results of the experiment to. (optional)
    """

    experiments = {}

    experiments[0] = ('Graph SVM Linear kernel accuracy vs. threshold',
                      save_svm_accuracy_threshold_graph,
                      ['results_directory','expression_file','ic50_file','thresholds','num_permutations'],
                      {'kernel' : 'linear'},
                      'linear_acc')

    experiments[1] = ('Graph SVM RBF kernel accuracy vs. threshold',
                      save_svm_accuracy_threshold_graph,
                      ['results_directory', 'expression_filename', 'ic50_filename', 'thresholds', 'num_permutations'],
                      {'kernel' : 'rbf'},
                      'rbf_acc')

    experiments[2] = ('Graph SVM Poly kernel accuracy vs. threshold',
                      save_svm_accuracy_threshold_graph,
                      ['results_directory', 'expression_filename', 'ic50_filename', 'thresholds', 'num_permutations'],
                      {'kernel' : 'poly'},
                      'poly_acc')

    experiments[3] = ('Graph all SVM kernel accuracies vs. threshold on same graph',
                      save_svm_accuracy_threshold_graph_multiple_kernels,
                      ['results_directory','linear_acc','rbf_acc','poly_acc'])

    experiments[4] = ('Save SVM Linear kernel model coefficients',
                      save_svm_model_coefficients,
                      ['results_directory','expression_filename','ic50_filename','thresholds'])

    experiments[5] = ('Save SVM Linear kernel patient predictions with undetermined cell lines',
                      save_svm_patient_predictions,
                      ['results_directory', 'expression_filename', 'ic50_filename','patient_directory', 'thresholds'],
                      {'kernel' : 'linear', 'trimmed' : False})

    experiments[6] = ('Save SVM RBF kernel patient predictions with undetermined cell lines',
                      save_svm_patient_predictions,
                      ['results_directory', 'expression_filename', 'ic50_filename','patient_directory', 'thresholds'],
                      {'kernel' : 'rbf', 'trimmed' : False})

    experiments[7] = ('Save SVM Poly kernel patient predictions with undetermined cell lines',
                      save_svm_patient_predictions,
                      ['results_directory', 'expression_filename', 'ic50_filename','patient_directory', 'thresholds'],
                      {'kernel' : 'poly', 'trimmed' : False})

    experiments[8] = ('Save SVM Linear kernel patient predictions without undetermined cell lines',
                      save_svm_patient_predictions,
                      ['results_directory', 'expression_filename', 'ic50_filename','patient_directory', 'thresholds'],
                      {'kernel' : 'linear', 'trimmed' : True})

    experiments[9] = ('Save SVM RBF kernel patient predictions without undetermined cell lines',
                      save_svm_patient_predictions,
                      ['results_directory', 'expression_filename', 'ic50_filename','patient_directory', 'thresholds'],
                      {'kernel' : 'rbf', 'trimmed' : True})

    experiments[10] = ('Save SVM Poly kernel patient predictions without undetermined cell lines',
                      save_svm_patient_predictions,
                      ['results_directory', 'expression_filename', 'ic50_filename','patient_directory', 'thresholds'],
                      {'kernel' : 'poly', 'trimmed' : True})

    experiments[11] = ('Save SVM Linear kernel full CCLE dataset predictions',
                       save_svm_full_CCLE_dataset_predictions,
                       ['results_directory', 'expression_filename', 'ic50_filename', 'thresholds'],
                       {'kernel' : 'linear'})

    experiments[12] = ('Save SVM RBF kernel full CCLE dataset predictions',
                       save_svm_full_CCLE_dataset_predictions,
                       ['results_directory', 'expression_filename', 'ic50_filename', 'thresholds'],
                       {'kernel' : 'rbf'})

    experiments[13] = ('Save SVM Poly kernel full CCLE dataset predictions',
                       save_svm_full_CCLE_dataset_predictions,
                       ['results_directory', 'expression_filename', 'ic50_filename', 'thresholds'],
                       {'kernel' : 'polynomial'})

    return experiments

def get_experiment_descriptions():
    experiments = define_experiments()
    return "\n".join(str(key) + " : " + experiments[key][0] for key in experiments.keys())


def default_parameters():
    parameters = {}
    parameters['experiments'] = [x for x in xrange(0,len(define_experiments()))]
    parameters['results_dir'] = os.path.dirname(__file__) + '/../Results/'
    parameters['expression_file'] = os.path.dirname(__file__) + '/../Data/CCLE_Data/sample1000.res'
    parameters['full_expression_file'] = os.path.dirname(__file__) + '/../Data/CCLE_Data/CCLE_Expression_2012-09-29.res'
    parameters['ic50_file'] = os.path.dirname(__file__) + '/../Data/IC_50_Data/CL_Sensitivity.txt'
    parameters['patient_dir'] = os.path.dirname(__file__) + "/../Data/TCGA_Data/9f2c84a7-c887-4cb5-b6e5-d38b00d678b1/Expression-Genes/UNC__AgilentG4502A_07_3/Level_3"
    parameters['threshold_increment'] = .01
    parameters['num_thresholds'] = 100
    parameters['num_permutations'] = 100
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
    os.mkdir(results_directory + "Predictions")

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

def save_svm_patient_predictions(results_directory,expression_file, ic50_file,patient_directory, thresholds,model_parameters={'kernel' : 'linear'},trimmed=False):
    results_file = results_directory + "Predictions/SVM_patient_prediction_%s_kernel_with%s_undetermined.txt" % (model_parameters['kernel'],"out" if trimmed else "")
    writer = open(results_file,"wb")
    for threshold in thresholds:
        writer.write("Threshold: %s\n" % str(threshold))
        model = classify.construct_svc_model(kernel="linear")
        identifiers,predictions = classify.get_svm_patient_predictions(model,expression_file,ic50_file,patient_directory,threshold)
        writer.write("\t".join(str(iden) for iden in identifiers) + "\n")
        writer.write("\t".join(str(pred) for pred in predictions)  + "\n\n")
    writer.close()

def save_svm_full_CCLE_dataset_predictions(results_directory,expression_file,ic50_file,thresholds,model_parameters={'kernel' : 'linear'}):
    results_file = results_directory + "Predictions/SVM_full_CCLE_predictions_%s_kernel.txt" % model_parameters['kernel']
    writer = open(results_file,"wb")
    for threshold in thresholds:
        writer.write("Threshold: %s\n" % str(threshold))
        model = classify.construct_svc_model(kernel="linear")
        cell_lines, predictions = classify.get_svm_predictions_full_dataset(model,expression_file,ic50_file,threshold)
        writer.write("\t".join(str(cell) for cell in cell_lines) + "\n")
        writer.write("\t".join(str(pred) for pred in predictions)  + "\n\n")
    writer.close()

def log(log_file, message):
    writer = open(log_file,"a+")
    writer.write(message)
    writer.close()

if __name__ == '__main__':
    main()