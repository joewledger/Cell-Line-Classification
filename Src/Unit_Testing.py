import SVM_Classification as svmc
import DataFormatting as dfm
import Plotting as plt
import Results as results
import pandas as pd

ic_50_filename = "IC_50_Data/CL_Sensitivity.txt"
expression_features_filename = "CCLE_Data/CCLE_Expression_2012-09-29.res"
#expression_features_filename = "CCLE_Data/sample1000.res"
tcga_dirctory = "TCGA_Data/9f2c84a7-c887-4cb5-b6e5-d38b00d678b1/Expression-Genes/UNC__AgilentG4502A_07_3/Level_3"

def test_cross_validation_feature_selection():
	outdir = "Results_CV/"
	results.make_dirs(outdir)
	thresholds = results.generate_thresholds(.01,.20)
	svm = svmc.SVM_Classification(ic_50_filename,expression_features_filename,exclude_undetermined=True,kernel='linear',thresholds=thresholds)
	all_predictions,all_features, all_evaluations = svm.evaluate_all_thresholds(5)
	results.write_cv_features(outdir,all_features,'linear')

def test_full_model_feature_selection():
	outdir = "Results_Full_Model/"
	results.make_dirs(outdir)
	thresholds = results.generate_thresholds(.01,.20)
	svm = svmc.SVM_Classification(ic_50_filename,expression_features_filename,exclude_undetermined=True,kernel='linear',thresholds=thresholds)
	full_model_predictions = svm.get_all_full_model_predictions()
	results.write_full_model_features(outdir,full_model_predictions,thresholds,'linear')

def test_matrix_normalization():
	df = dfm.DataFormatting(ic_50_filename ,expression_features_filename,tcga_dirctory)
	expression_matrix = df.generate_cell_line_expression_matrix()
	print(df.normalize_expression_matrix(expression_matrix))

def test_matrix_generation():
	df = dfm.DataFormatting(ic_50_filename,expression_features_filename,None)
	print(df.generate_cell_line_expression_matrix())

test_matrix_generation()