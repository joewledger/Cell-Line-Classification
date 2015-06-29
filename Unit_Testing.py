import SVM_Classification as svmc
import DataFormatting as dfm
import Plotting as plt
import Results as results
import pandas as pd


def test_full_model_feature_selection():
	outdir = "Results_Full_Model/"
	make_dirs(outdir)
	thresholds = generate_thresholds(.01,.20)
	df = dfm.DataFormatting(ic_50_filename ,expression_features_filename,tcga_dirctory)
	svm = svmc.SVM_Classification(df,exclude_undetermined=True,kernel='linear',thresholds=thresholds)
	full_model_predictions = svm.get_all_full_model_predictions()
	results.write_full_model_features(outdir,full_model_predictions,thresholds,'linear')

def test_matrix_normalization():
	ic_50_filename = "IC_50_Data/CL_Sensitivity.txt"
	expression_features_filename = "CCLE_Data/CCLE_Expression_2012-09-29.res"
	#expression_features_filename = "CCLE_Data/sample100.res"
	tcga_dirctory = "TCGA_Data/9f2c84a7-c887-4cb5-b6e5-d38b00d678b1/Expression-Genes/UNC__AgilentG4502A_07_3/Level_3"
	df = dfm.DataFormatting(ic_50_filename ,expression_features_filename,tcga_dirctory)
	expression_matrix = df.generate_cell_line_expression_matrix()
	print(df.normalize_expression_matrix(expression_matrix))

test_matrix_normalization()