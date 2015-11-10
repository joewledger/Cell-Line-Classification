import argparse
from argparse import RawDescriptionHelpFormatter
import datetime
import os
import DataFormatter as dfm
import Classification as classify
from multiprocessing import Pool
import itertools as iter
import traceback


def main():
    description = """
    A tool for predicting patient response to cancer drugs using several different machine learning models.\n\n
    These include Support Vector Machines, Artificial Neural Networks, and the NEAT algorithm.\n\n
    Depending on the parameters given, this module will run different experiments.\n\n""" + get_experiment_descriptions()
    description = description.replace("    ","")
    description += get_default_parameter_descriptions()

    parser = argparse.ArgumentParser(description=description,formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--experiments',nargs='+',type=int,help='The experiments to run.')
    parser.add_argument('--results_dir',type=str,help='The directory to save results to.')
    parser.add_argument('--expression_file',type=str,help='The file to read gene expression measurements from (Use \'full\' for full dataset).')
    parser.add_argument('--ic50_file',type=str,help='The file to read the IC50 measurements from.')
    parser.add_argument('--patient_dir',type=str,help='The directory containing the patient information')
    parser.add_argument('--threshold_increment',type=float,help='The increment to test thresholds at.')
    parser.add_argument('--num_thresholds',type=int,help='The number of thresholds to test.')
    parser.add_argument('--num_features_increment',type=float,help='The increment to test num_features at.')
    parser.add_argument('--num_feature_sizes_to_test',type=int,help='The number of feature sizes to test.')
    parser.add_argument('--num_permutations',type=int,help='The number of permutations to use for cross-fold validation.')
    parser.add_argument('--num_threads', type=int,help='The number of threads to use for multiproccesing (if supported by experiment')
    parser.add_argument('--target_features', type=int,help='The number of features to target in bidirectional feature search')
    parser.add_argument('--model',type=str,help='The type of model to use. Options are \'svm\', \'nn\', and \'dt\'')
    parser.add_argument('--kernel',type=str,help='The SVM kernel type to use. Options are \'linear\', \'rbf\', or \'poly\'')
    parser.add_argument('--trimmed',type=bool,help='Whether or not to exclude undetermined cell lines when training model for patient predictions.')
    parser.add_argument('--drug',type=str,help="The drug to use IC50 measurements for")
    parser.set_defaults(**default_parameters())

    args = parser.parse_args()
    params = configure_parameters(args)

    experiment_definitions = define_experiments()
    experiments = [key for key in experiment_definitions.keys() if key in params['experiments']]
    run_experiments(experiments,params)

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

    experiments[0] = ('Write accuracy v. threshold scores to text file',
                      write_accuracy_threshold_scores_to_file,
                      ['results_dir','model_object','expression_file','ic50_file','thresholds','num_permutations','drug','num_threads'],
                      ['kernel'])

    experiments[1] = ('Write accuracy v. #features scores to text file',
                      write_accuracy_features_scores_to_file,
                      ['results_dir','model_object','expression_file','ic50_file','feature_sizes','num_permutations','num_threads'],
                      ['kernel'])

    experiments[2] = ('Write full CCLE predictions to file',
                      write_full_CCLE_predictions_to_file,
                      ['results_dir','model_object','expression_file','ic50_file','thresholds'],
                      ['kernel'])

    experiments[3] = ('Write SVM model coefficients to file',
                      write_svm_model_coefficients_to_file,
                      ['results_dir','expression_file','ic50_file','thresholds'],
                      ['kernel'])

    experiments[4] = ('Write Patient Predictions to file',
                      write_patient_predictions_to_file,
                      ['results_dir','model_object','expression_file','ic50_file','patient_dir','thresholds'],
                      ['kernel','trimmed'])

    experiments[5] = ('Write SVM Model Accuracy bidirectional feature search accuracy scores to file',
                      write_svm_model_accuracy_bidirectional_feature_search,
                      ['results_dir','expression_file','ic50_file','target_features','num_permutations'],
                      ['kernel'])

    experiments[6] = ('Write SVM Model Accuracy RFE accuracy scores to file',
                      write_RFE_accuracy_features_scores_to_file,
                      ['results_dir','model_object','expression_file','ic50_file','feature_sizes','num_permutations','num_threads'],
                      ['kernel'])

    experiments[7] = ('Write full CCLE predictions top features to file',
                      write_full_CCLE_predictions_top_features_to_file,
                      ['results_dir','model_object','expression_file','ic50_file','target_features'],
                      ['kernel'])

    experiments[8] = ('Write patient predictions top features to file',
                      write_patient_predictions_to_file_top_features,
                      ['results_dir','model_object','expression_file','ic50_file','patient_dir','target_features'],
                      ['kernel','trimmed'])

    return experiments

def default_parameters():
    parameters = {}
    parameters['experiments'] = [x for x in xrange(0,len(define_experiments()))]
    parameters['results_dir'] = os.path.dirname(__file__) + '/../Results/'
    parameters['expression_file'] = os.path.dirname(__file__) + '/../Data/CCLE_Data/sample100.res'
    parameters['full_expression_file'] = os.path.dirname(__file__) + '/../Data/CCLE_Data/CCLE_Expression_2012-09-29.res'
    parameters['ic50_file'] = os.path.dirname(__file__) + '/../Data/IC_50_Data/CL_Sensitivity_Multiple_Drugs.csv'
    parameters['patient_dir'] = os.path.dirname(__file__) + "/../Data/TCGA_Data/9f2c84a7-c887-4cb5-b6e5-d38b00d678b1/Expression-Genes/UNC__AgilentG4502A_07_3/Level_3"
    parameters['threshold_increment'] = .01
    parameters['num_thresholds'] = 100
    parameters['num_permutations'] = 100
    parameters['num_features_increment'] = 5
    parameters['num_feature_sizes_to_test'] = 10
    parameters['num_threads'] = 5
    parameters['target_features'] = 5
    parameters['model'] = 'svm'
    parameters['kernel'] = 'linear'
    parameters['trimmed'] = True
    parameters['drug'] = "SMAP"
    return parameters

def get_default_parameter_descriptions():
    return "\n\nDefault Parameters:\n" + "\n".join(str(parameter)
           + " : " + str(default_parameters()[parameter]) for parameter in default_parameters().keys())

def configure_parameters(args):
    params = dict(vars(args))

    params['results_dir'] = get_results_filepath(params['results_dir'])
    make_results_dir_and_subdirectories(args.results_dir,params['results_dir'])

    if params['expression_file'] == "full":
        params['expression_file'] = params['full_expression_file']

    params['thresholds'] = [params['threshold_increment'] * x for x in xrange(1,params['num_thresholds'] + 1)]
    params['feature_sizes'] = [params['num_features_increment'] * x for x in xrange(1,params['num_feature_sizes_to_test'] + 1)]

    models = {'svm' : classify.SVM_Model(), 'nn' : classify.Neural_Network_Model(), 'dt' : classify.Decision_Tree_Model()}

    params['model_object'] = models[params['model']]
    return params

def run_experiments(experiments, params):
    experiment_definitions = define_experiments()
    log_file = params['results_dir'] + "log.txt"
    exp_des = "Experiment parameters\n" + "\n".join("%s: %s" % (p[0],p[1]) for p in params.items()) + "\n\n"
    log(log_file,exp_des)


    for experiment in experiments:
        curr_exp = experiment_definitions[experiment]
        experiment_description = "%s (%s)" % (str(experiment),curr_exp[0])
        log(log_file, "Starting Experiment %s at %s\n" % (experiment_description, str(datetime.datetime.today())))

        try:
            method = curr_exp[1]
            args = [params[x] for x in curr_exp[2]]
            kwargs = {x : params[x] for x in curr_exp[3]} if len(curr_exp) > 3 else {}
            if(not params['model'] == 'svm'):
                kwargs.pop('kernel')
            save_var = curr_exp[4] if len(curr_exp) > 4 else None

            if save_var:
                params[save_var] = experiment_wrapper(method,args,kwargs)
            else:
                experiment_wrapper(method,args,kwargs)

            log(log_file, "Finished Experiment %s at %s\n" % (experiment_description, str(datetime.datetime.today())))
        except Exception, e:
            log(log_file, "Experiment %s failed at %s\n" % (experiment_description, str(datetime.datetime.today())))
            log(log_file, "\t%s" % str(traceback.format_exc()))

def experiment_wrapper(func,args,kwargs):
    return func(*args,**kwargs)

def get_experiment_descriptions():
    experiments = define_experiments()
    return "\n".join(str(key) + " : " + experiments[key][0] for key in experiments.keys())

def get_results_filepath(base_results_dir):
    """
    Gets the filepath for a new results directory based on the current date and time
    """
    curr_time = str(datetime.datetime.today()).replace(" ","_").replace(":","-")
    curr_time = curr_time[:curr_time.rfind("-")]
    return base_results_dir + curr_time + "/"

def make_results_dir_and_subdirectories(base_results_dir,results_dir):

    if not os.path.isdir(base_results_dir):
        os.mkdir(base_results_dir)
    os.mkdir(results_dir)
    os.mkdir(results_dir + "Plots")
    os.mkdir(results_dir + "Plots/SVM_Accuracies")
    os.mkdir(results_dir + "Plots/Decision_Tree_Accuracies")
    os.mkdir(results_dir + "Model_Coefficients")
    os.mkdir(results_dir + "Predictions")
    os.mkdir(results_dir + "Accuracy_Scores")

def map_wrapper(all_args):
    func = all_args[0]
    args = all_args[1:-1]
    kwargs = all_args[-1]
    func(*args,**kwargs)

def write_accuracy_threshold_scores_to_file(results_dir,model_object,expression_file,ic50_file,thresholds,num_permutations,drug,num_threads,**kwargs):
    pool = Pool(num_threads)
    pool.map(map_wrapper,
             iter.izip(iter.repeat(_write_accuracy_threshold),
                    iter.repeat(results_dir),
                    iter.repeat(model_object),
                    iter.repeat(expression_file),
                    iter.repeat(ic50_file),
                    thresholds,
                    iter.repeat(num_permutations),
                    iter.repeat(drug),
                    iter.repeat(kwargs)))

def _write_accuracy_threshold(results_dir,model_object, expression_file,ic50_file,threshold,num_permutations,drug,**kwargs):

    savefile = results_dir + "Accuracy_Scores/SVM_%s_accuracy_%s_threshold.txt" % (kwargs['kernel'] , str(threshold))
    accuracy_scores = model_object.get_model_accuracy_filter_threshold(expression_file,ic50_file,threshold,num_permutations,drug,**kwargs)
    writer = open(savefile,"wb")
    for value in accuracy_scores:
        writer.write(str(value) + "\n")
    writer.close()

def write_accuracy_features_scores_to_file(results_dir,model_object,expression_file,ic50_file,feature_sizes,num_permutations,num_threads,**kwargs):
    pool = Pool(num_threads)
    pool.map(map_wrapper,
             iter.izip(iter.repeat(_write_accuracy_features),
                    iter.repeat(results_dir),
                    iter.repeat(model_object),
                    iter.repeat(expression_file),
                    iter.repeat(ic50_file),
                    feature_sizes,
                    iter.repeat(num_permutations),
                    iter.repeat(kwargs)))

def _write_accuracy_features(results_dir,model_object, expression_file,ic50_file,feature_size,num_permutations,**kwargs):

    savefile = results_dir + "Accuracy_Scores/SVM_%s_accuracy_%s_features.txt" % (kwargs['kernel'] , str(int(feature_size)))
    accuracy_scores = model_object.get_model_accuracy_filter_feature_size(expression_file,ic50_file,int(feature_size),num_permutations,**kwargs)
    writer = open(savefile,"wb")
    writer.write("Number of Features: %s\n" % str(feature_size))
    writer.close()
    for value in accuracy_scores:
        writer = open(savefile,"a")
        writer.write(str(value) + "\n")
        writer.close()

def write_RFE_accuracy_features_scores_to_file(results_dir,model_object,expression_file,ic50_file,feature_sizes,num_permutations,num_threads,**kwargs):
    pool = Pool(num_threads)
    pool.map(map_wrapper,
             iter.izip(iter.repeat(_write_RFE_accuracy_features),
                    iter.repeat(results_dir),
                    iter.repeat(model_object),
                    iter.repeat(expression_file),
                    iter.repeat(ic50_file),
                    feature_sizes,
                    iter.repeat(num_permutations),
                    iter.repeat(kwargs)))

def _write_RFE_accuracy_features(results_dir,model_object, expression_file,ic50_file,feature_size,num_permutations,**kwargs):

    savefile = results_dir + "Accuracy_Scores/SVM_%s_RFE_accuracy_%s_features.txt" % (kwargs['kernel'] , str(int(feature_size)))
    accuracy_scores = model_object.get_model_accuracy_RFE(expression_file,ic50_file,int(feature_size),num_permutations,**kwargs)
    writer = open(savefile,"wb")
    writer.write("Number of Features: %s\n" % str(feature_size))
    writer.close()
    for value in accuracy_scores:
        writer = open(savefile,"a")
        writer.write(str(value) + "\n")
        writer.close()

def write_full_CCLE_predictions_to_file(results_dir,model_object,expression_file,ic50_file,thresholds,**kwargs):
    results_file = results_dir + "Predictions/SVM_full_CCLE_predictions_%s_kernel.txt" % kwargs['kernel']
    writer = open(results_file,"wb")
    for threshold in thresholds:
        writer.write("Threshold: %s\n" % str(threshold))
        cell_lines, predictions = model_object.get_predictions_full_CCLE_dataset(expression_file,ic50_file,threshold,**kwargs)
        writer.write("\t".join(str(cell) for cell in cell_lines) + "\n")
        writer.write("\t".join(str(pred) for pred in predictions)  + "\n\n")
    writer.close()

def write_full_CCLE_predictions_top_features_to_file(results_dir,model_object,expression_file,ic50_file,target_features,**kwargs):
    results_file = results_dir + "Predictions/SVM_full_CCLE_predictions_%s_kernel.txt" % kwargs['kernel']
    writer = open(results_file,"wb")
    cell_lines, predictions,top_features = model_object.get_predictions_full_CCLE_dataset_from_top_features(expression_file,ic50_file,target_features,**kwargs)
    writer.write("Top Features: %s\n" % "\t".join(str(x) for x in top_features))
    writer.write("\t".join(str(cell) for cell in cell_lines) + "\n")
    writer.write("\t".join(str(pred) for pred in predictions)  + "\n\n")
    writer.close()

def write_svm_model_coefficients_to_file(results_dir,expression_file,ic50_file,thresholds,**kwargs):
    results_file = results_dir + "Model_Coefficients/svm_linear.txt"
    writer = open(results_file,"wb")
    for threshold in thresholds:
        writer.write("Threshold: %s\n" % str(threshold))
        model = classify.SVM_Model()
        expression_frame,ic50_series = dfm.get_expression_frame_and_ic50_series(expression_file, ic50_file,normalized=True,trimmed=True,threshold=threshold)
        genes = list(expression_frame.index)
        model_coefficients = model.get_model_coefficients(expression_file,ic50_file,threshold,**kwargs)
        writer.wr_fite("\t".join(str(gene) for gene in genes) + "\n")
        writer.write("\t".join(str(coef) for coef in model_coefficients)  + "\n\n")
    writer.close()

def write_patient_predictions_to_file(results_dir,model_object,expression_file,ic50_file,patient_dir,thresholds,**kwargs):
    results_file = results_dir + "Predictions/SVM_patient_prediction_%s_kernel_with%s_undetermined.txt" % (kwargs['kernel'],"out" if kwargs['trimmed'] else "")
    writer = open(results_file,"wb")
    for threshold in thresholds:
        writer.write("Threshold: %s\n" % str(threshold))
        identifiers,predictions = model_object.get_patient_predictions(expression_file,ic50_file,patient_dir,threshold)
        writer.write("\t".join(str(iden) for iden in identifiers) + "\n")
        writer.write("\t".join(str(pred) for pred in predictions)  + "\n\n")
    writer.close()

def write_patient_predictions_to_file_top_features(results_dir,model_object,expression_file,ic50_file,patient_dir,target_features,**kwargs):
    results_file = results_dir + "Predictions/SVM_patient_prediction_%s_kernel_with%s_undetermined.txt" % (kwargs['kernel'],"out" if kwargs['trimmed'] else "")
    writer = open(results_file,"wb")
    identifiers,predictions,top_features = model_object.get_patient_predictions_top_features(expression_file,ic50_file,patient_dir,target_features,trimmed=kwargs['trimmed'])
    writer.write("Top Features: %s\n" % "\t".join(str(x) for x in top_features))
    writer.write("\t".join(str(iden) for iden in identifiers) + "\n")
    writer.write("\t".join(str(pred) for pred in predictions)  + "\n\n")
    writer.close()

def write_svm_model_accuracy_bidirectional_feature_search(results_dir,expression_file,ic50_file,target_features,num_permutations,**kwargs):
    model = classify.SVM_Model()
    accuracy_scores = model.get_model_accuracy_bidirectional_feature_search(expression_file,ic50_file,target_features,num_permutations,kernel=kwargs['kernel'])
    results_file = results_dir + "Accuracy_Scores/SVM_bidirectional_%s_features.txt" % str(target_features)
    writer = open(results_file,"wb")
    writer.write("Target Features: %s\n" % str(target_features))
    writer.close()

    for score in accuracy_scores:
        writer = open(results_file,"a")
        writer.write(str(score) + "\n")
        writer.close()


def log(log_file, message):
    writer = open(log_file,"a+")
    writer.write(message)
    writer.close()

if __name__ == '__main__':
    main()