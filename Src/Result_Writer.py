import datetime
import os

"""
Writes results to a results directory

"""
base_results_directory = os.path.dirname(__file__) + '/../Results/'

def get_results_filepath():
    """
    Gets the filepath for a new results directory based on the current date and time
    """
    curr_time = str(datetime.datetime.today()).replace(" ","_").replace(":","-")
    curr_time = curr_time[:curr_time.rfind("-")]
    return base_results_directory + curr_time

def make_results_directory_and_subdirectories(results_directory):
    if not os.path.isdir(base_results_directory):
        os.mkdir(base_results_directory)
    os.mkdir(results_directory)
    os.mkdir(results_directory + "/Plots")
    os.mkdir(results_directory + "/Model_Coefficients")





results_directory = get_results_filepath()
make_results_directory_and_subdirectories(results_directory)




