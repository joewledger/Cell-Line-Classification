import SVM_Classification as svmc
import DataFormatting as dfm
import Plotting as plt
import Results as results


def test_full_model_feature_selection():
	outdir = "Results_Full_Model/"
	make_dirs(outdir)
	thresholds = generate_thresholds(.01,.20)
	df = dfm.DataFormatting(ic_50_filename ,expression_features_filename,tcga_dirctory)
	svm = svmc.SVM_Classification(df,exclude_undetermined=True,kernel='linear',thresholds=thresholds)
	full_model_predictions = svm.get_all_full_model_predictions()
	results.write_full_model_features(outdir,full_model_predictions,thresholds,'linear')