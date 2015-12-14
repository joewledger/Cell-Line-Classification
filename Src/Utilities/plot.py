import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import os
import sys
import numpy as np

def plot_accuracy_multiple_models(title,xlabel,data_dir,outfile,trim_axis=False):
	full_path = "%s/%s" % (os.getcwd(),data_dir)
	outfile = "%s/%s" % (os.getcwd(), outfile)

	folders = [x[0] for x in os.walk(full_path)][1:]
	scores = map_folders_to_accuracy_scores(full_path,folders)
	plot_scores(title,xlabel,outfile,scores)

def plot_scores(title,xlabel,outfile,scores):
	plt.figure()
	plt.xlabel(xlabel)
	plt.ylabel("Accuracy")
	plt.title(title)

	colors = ['red','blue','green']
	model_names = [parse_model_name(folder) for folder in scores.keys()]

	for i,folder in enumerate(scores.keys()):
		model_name = parse_model_name(folder)
		thresholds = sorted(scores[folder].keys())
		accuracy_means = [scores[folder][threshold][0] for threshold in thresholds]
		accuracy_std = [scores[folder][threshold][1] for threshold in thresholds]
		plt.plot(thresholds,accuracy_means,label=model_names[(i-1) % len(model_names)],color=colors[i])
		plt.errorbar(thresholds, accuracy_means,yerr=accuracy_std)
	plt.legend()
	plt.savefig(outfile)
	plt.close()


def map_folders_to_accuracy_scores(full_path,folders):
	scores = {}
	for folder in folders:
		scores[folder] = {}
		files = [f for f in os.listdir(folder) if f.endswith(".txt")]
		for f in files:
			threshold = parse_threshold(f)
			curr_file = full_path + parse_model_name(folder) + "/" + f
			reader = open(curr_file,"rb")
			threshold_scores = np.array([float(line) for line in reader.readlines()])
			reader.close()
			scores[folder][threshold] = (threshold_scores.mean(),threshold_scores.std())
	return scores

def parse_model_name(folder):
	return folder[folder.rfind("/") + 1:]

def parse_threshold(filename):
	try:
		return float(filename[filename.find("_") + 1: filename.rfind("_")])
	except:
		begin_str = "RFE_Accuracy"
		return int(filename[len(begin_str) + 1:filename.rfind("_")])


if __name__ == "__main__":
	#title,xlabel,data_dir,outfile
	plot_accuracy_multiple_models(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],trim_axis=True)
