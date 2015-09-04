import datetime
import os
import DataFormatter as dfm
import Classification as classify
import Plotting as plt

"""
Writes results to a results directory

"""
base_results_directory = os.path.dirname(__file__) + '/../Results/'
expression_filename = os.path.dirname(__file__) + '/../Data/CCLE_Data/sample1000.res'
ic50_filename = os.path.dirname(__file__) + '/../Data/IC_50_Data/CL_Sensitivity.txt'

def main():
    results_directory = get_results_filepath()
    make_results_directory_and_subdirectories(results_directory)
    save_accuracy_threshold_graph(results_directory, expression_filename,ic50_filename,[float(x) * .01 for x in xrange(1,101)])

def get_results_filepath():
    """
    Gets the filepath for a new results directory based on the current date and time
    """
    curr_time = str(datetime.datetime.today()).replace(" ","_").replace(":","-")
    curr_time = curr_time[:curr_time.rfind("-")]
    return base_results_directory + curr_time + "/"

def make_results_directory_and_subdirectories(results_directory):
    if not os.path.isdir(base_results_directory):
        os.mkdir(base_results_directory)
    os.mkdir(results_directory)
    os.mkdir(results_directory + "Plots")
    os.mkdir(results_directory + "Model_Coefficients")

def save_svm_accuracy_threshold_graph(results_directory,expression_file,ic50_file,thresholds):
    model = classify.construct_svc_model(kernel="linear")
    accuracies = classify.get_svm_model_accuracy_multiple_thresholds(model, expression_file, ic50_file, thresholds)
    outfile = results_directory + "Plots/accuracy_threshold.png"
    plt.plot_accuracy_threshold_curve(outfile,[x[0] for x in accuracies],[x[1] for x in accuracies])


if __name__ == '__main__':
    main()










