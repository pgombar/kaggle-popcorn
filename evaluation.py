# Models has to have a .fit(X, Y)  function,
# 				 and a .predict(X) function
# Features has to have a .fit_transform function,
#				   and a .transform     function
#				   and a .name			string

from preprocessing   import get_train_data, get_test_data, \
							Get_Testing_Data, Write_Predictions
from sklearn.metrics import roc_auc_score
import time, sys
	
def rprint(str): # Next print overwrites this, eg use for indicate progress
	sys.stdout.write("  Model evaluation : " + str + "            \r")
	sys.stdout.flush()

def evaluate(y_true, y_predicted):
	# Evaluation metric is the area under the ROC-curve.
	return roc_auc_score(y_true, y_predicted)


# Fits model, transforms features, evaluates results, everything
def evaluate_model_feature(model, feature, ratio = 0.7):
	start = time.time();
	model_name, feature_name = str(model), feature.name;
	print "==========================================================="
	print '\033[1mModel:   \033[07m\033[04m' + model_name   + '\033[0m'
	print '\033[1mFeature: \033[07m\033[04m' + feature_name + '\033[0m'
	
# Model fitting
	X_train, Y_train = get_train_data(ratio)
	rprint('Feature extraction of train data')
	train_features = feature.fit_transform(X_train)
	del X_train								#a lot of memory
	rprint('Fitting model                           ')
	if str(model)[0:16] == "Semi-supervised ":
		model.feature = feature
	model.fit(train_features, Y_train)		# may use too much ram
	
# Test on train data
	rprint('Creating predictions for train data     ')
	predictions_train = model.predict(train_features)
	del train_features;						#a lot of memory
	rprint('Evaluating predictions for train data')
	score_train = evaluate(Y_train, predictions_train)
	del predictions_train;	del Y_train;	#not much memory
	
	if ratio < 1:
# Testing on the test data
		X_test,  Y_test  = get_test_data(ratio)
		rprint('Feature extraction of test data')
		test_features  = feature.transform(X_test)
		del X_test; del feature;			#not much memory
		rprint('Creating predictions for test data')
		predictions_test  = model.predict(test_features)
		del test_features; del model;			#maybe a lot of memory
		rprint('Evaluating predictions for test data')
		score_test = evaluate(Y_test, predictions_test)
		del predictions_test;	del Y_test;		#not much memory
	else:
# Generating output csv for ratio = 1
		Data, IDs = Get_Testing_Data()
		rprint('Feature extraction of Kaggle test data')
		Data_features = feature.transform(Data)
		del Data; del feature;
		rprint('Creating predictions for Kaggle test data')
		Predictions = model.predict(Data_features)
		del Data_features; del model;
		Write_Predictions("output.csv", IDs, Predictions)
#Printing results
	sys.stdout.write("                                                            \r")
	print "Scores:\t (runtime = %0.3f sec)" % (time.time() - start)
	print "\t %f on train set" % score_train
	if ratio < 1:
		print "\t\033[01m %f" % score_test + "\033[0m on test set"
	else:
		print "\t Unknown real life score"