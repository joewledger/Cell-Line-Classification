
import DataFormatter as dfm
import Cross_Validator as cv
import os
import Src.NEAT.NeatClassifier as n
import matplotlib.pyplot as plt
import numpy as np
import datetime
from multiprocessing import Pool

def get_accuracy_and_runtime_vs_num_generations(expression_file,ic50_file,num_features,generation_range,num_permutations,num_threads):
    drug = "SMAP"

    scikit_data,scikit_target = dfm.get_expression_scikit_data_target_for_drug(expression_file,ic50_file,drug,normalized=True,trimmed=True,threshold=None)
    p = Pool(num_threads)
    scores = p.map(wrap, [(g,scikit_data,scikit_target,num_features,num_permutations) for g in generation_range])
    scores = {g : scores[i] for i,g in enumerate(generation_range)}
    return scores

def wrap(args):
    return acc_and_run(*args)

def acc_and_run(g,scikit_data,scikit_target,num_features,num_permutations):
    for perm in xrange(0,num_permutations):
        try:
            start_time = datetime.datetime.now()
            model = n.NeatClassifier(max_generations=g,config_file='NEAT/config.txt')
            shuffled_data,shuffled_target = dfm.shuffle_scikit_data_target(scikit_data,scikit_target)
            acc = cv.cross_val_score_filter_feature_selection(model,cv.trim_X_num_features,num_features,shuffled_data,shuffled_target,cv=5)
            end_time = datetime.datetime.now()
            return acc.mean(),(end_time - start_time).seconds
        except:
            return 0.0, 1000.0

def plot_accuracy_num_generations(savefile,accuracy_scores):
    plt.figure()
    plt.xlabel("Number of Generations")
    plt.ylabel("Accuracy")
    plt.title("NEAT accuracy vs. Number of Generations")

    x_pts = sorted(accuracy_scores.keys())
    y_pts = [np.array(accuracy_scores[key][0]).mean() for key in x_pts]
    y_err = [np.array(accuracy_scores[key][0]).std() for key in x_pts]
    plt.plot(x_pts,y_pts)
    plt.errorbar(x_pts,y_pts,yerr=y_err)
    plt.savefig(savefile)
    plt.close()

def plot_runtime_num_generations(savefile,runtime_scores):
    plt.figure()
    plt.xlabel("Number of Generations")
    plt.ylabel("Runtime per Permutation (seconds)")
    plt.title("NEAT runtime vs. Number of Generations")

    x_pts = sorted(runtime_scores.keys())
    y_pts = [np.array(runtime_scores[key][1]).mean() for key in x_pts]
    y_err = [np.array(runtime_scores[key][1]).std() for key in x_pts]
    plt.plot(x_pts,y_pts)
    plt.errorbar(x_pts,y_pts,yerr=y_err)
    plt.savefig(savefile)
    plt.close()

#Num_Permutations, Num_Threads
if __name__ == "__main__":
    expression_file = os.path.dirname(__file__) + '/../' + 'Data/CCLE_Data/full_expression.csv'
    ic_50_file = os.path.dirname(__file__) + '/../' + 'Data/IC_50_Data/CL_Sensitivity_Multiple_Drugs.csv'
    num_features = 10
    generation_range = [x * 5 for x in xrange(1,16)]
    num_permutations = sys.argv[1]
    num_threads = sys.argv[2]
    save_1 = os.path.dirname(__file__) + '/../' +  "Results/accuracy_num_generations.png"
    save_2 = os.path.dirname(__file__) + '/../' +  "Results/runtime_num_generations.png"

    accuracy_and_runtime = get_accuracy_and_runtime_vs_num_generations(expression_file,ic_50_file,num_features,generation_range,num_permutations,num_threads)

    plot_accuracy_num_generations(save_1,accuracy_and_runtime)
    plot_runtime_num_generations(save_2,accuracy_and_runtime)




