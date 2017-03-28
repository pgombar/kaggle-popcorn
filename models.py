from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from preprocessing import get_test_data, get_train_data
from features import *
import time
import sys

# set of models
#		   key				model name				model
models = {"mnb" :	("Multinomial Naive Bayes",	MultinomialNB()), \
		  "bnb":	("Bernoulli Naive Bayes",	BernoulliNB())}
# set of feature extractors
#				key				name				fit transform	transform
features = {"tftidf":	("Term-Frequency times Inverse Document-Frequency", tf_idf_fit_transform, tf_idf_transform), \
			"bow":		("Bag Of Words" , bow_fit_transform, bow_transform)}

def main():
	#test('mnb','bow')
	test_all()
	
def rprint(str): # Next print overwrites this, eg use for indicate progress
	sys.stdout.write("Model evaluation : " + str + "            \r")
	sys.stdout.flush()

def evaluate(y_true, y_predicted):
	# Evaluation metric is the area under the ROC-curve.
	return roc_auc_score(y_true, y_predicted)

# Fits model, transforms features, evaluates results
def evaluate_model_feature(model_name, model, feature_name, \
							feature_fit_transform, feature_transform):
	print 'Testing ' + model_name + ' with ' + feature_name
	start = time.time()
	
	# Model fitting
	X_train, Y_train = get_train_data();
	rprint('Feature extraction of train data')
	train_features, feature_model = feature_fit_transform(X_train);
	del X_train								#a lot of memory
	rprint('Fitting ' + model_name)
	model.fit(train_features, Y_train)		# may use too much ram
	
	# Test on train data
	rprint('Evaluating train data')
	predictions_train = model.predict(train_features)
	del train_features;						#a lot of memory
	score_train = evaluate(Y_train, predictions_train)
	del predictions_train;	del Y_train;	#not much memory
	
	# Testing on the test data
	rprint('Evaluating test data');
	X_test,  Y_test  = get_test_data()
	test_features  = feature_transform(X_test, feature_model)
	del X_test; del feature_model;			#not much memory
	predictions_test  = model.predict(test_features)
	del test_features; del model;			#maybe a lot of memory
	score_test = evaluate(Y_test, predictions_test)
	del predictions_test;	del Y_test;		#not much memory
	print "{0} - {1} scores:".format(model_name, feature_name)
	print "\t{0} on train set\n\t{1} on test set".format(score_train, score_test)
	print "Testing took \t{0} seconds.          ".format(time.time() - start)

def test_all():
	for model_key in models:
		model = models[model_key]
		for feature_key in features:
			feature = features[feature_key]
			evaluate_model_feature(model[0], model[1], feature[0], feature[1], feature[2])

def test(model_key, feature_key):
	model = models[model_key]
	feature = features[feature_key]
	evaluate_model_feature(model[0], model[1], feature[0], feature[1], feature[2])


if __name__ == "__main__":
	main()