# Models has to have a .fit(X, Y)  function,
# 				 and a .predict(X) function
# Features has to have a .fit_transform function,
#				   and a .transform     function

from preprocessing import get_train_data, get_test_data, \
							Get_Testing_Data, Write_Predictions
from sklearn.metrics import roc_auc_score
import time, sys
	
def rprint(str): # Next print overwrites this, eg use for indicate progress
	sys.stdout.write("Model evaluation : " + str + "            \r")
	sys.stdout.flush()

def evaluate(y_true, y_predicted):
	# Evaluation metric is the area under the ROC-curve.
	return roc_auc_score(y_true, y_predicted)

# Fits model, transforms features, evaluates results
def evaluate_model_feature(model_name, model, feature_name, feature, ratio = 0.7):
	print 'Testing ' + model_name + ' with ' + feature_name
	start = time.time()
	
# Model fitting
	X_train, Y_train = get_train_data(ratio);
	rprint('Feature extraction of train data')
	train_features = feature.fit_transform(X_train);
	del X_train								#a lot of memory
	rprint('Fitting ' + model_name)
	model.fit(train_features, Y_train)		# may use too much ram
	
# Test on train data
	rprint('Evaluating train data')
	predictions_train = model.predict(train_features)
	del train_features;						#a lot of memory
	score_train = evaluate(Y_train, predictions_train)
	del predictions_train;	del Y_train;	#not much memory
	
	if ratio < 1:
# Testing on the test data
		rprint('Evaluating test data');
		X_test,  Y_test  = get_test_data(ratio)
		test_features  = feature.transform(X_test)
		del X_test; del feature;			#not much memory
		predictions_test  = model.predict(test_features)
		del test_features; del model;			#maybe a lot of memory
		score_test = evaluate(Y_test, predictions_test)
		del predictions_test;	del Y_test;		#not much memory
	else:
# Generating output csv for ratio = 1
		Data, IDs = Get_Testing_Data()
		Data_features = feature.transform(Data)
		del Data; del feature;
		Predictions = model.predict(Data_features)
		del Data_features; del model;
		Write_Predictions("output.csv", IDs, Predictions)
#Printing results
	print "{0} - {1} scores:".format(model_name, feature_name)
	print "\t{0} on train set".format(score_train)
	if ratio < 1:
		print "\t{0} on test  set".format(score_test)
	else:
		print "\tUnknown real life score"
	print "Testing took \t{0} seconds.          ".format(time.time() - start)