from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from preprocessing import get_train_test_sets
from features import *
import time


def fit_and_predict(model, train_features, y_train, test_features):
	# Since the models from sklearn adhere to the same interface, we can pass the models as a function parameter and write generic code.
	# This function fits the model to the training data and predicts already seen data (train set) and unseen data (test set). We don't actually need prediction for the train set, this is just for comparison to the test set scores.

	model.fit(train_features, y_train)
	predictions_train = model.predict(train_features)
	predictions_test = model.predict(test_features)

	return predictions_train, predictions_test


def evaluate(y_true, y_predicted):
	# Evaluation metric is the area under the ROC-curve.
	return roc_auc_score(y_true, y_predicted)


def main():
	print "Starting..."
	start = time.time()

	# Preprocess data (clean and split into train and test set).
	X_train, X_test, y_train, y_test = get_train_test_sets()
	print "Done preprocessing data!"

	# Construct feature vector. Replace tf_idf with any other function from features.py.
	train_features, test_features = tf_idf(X_train, y_train, X_test)
	print "Done constructing features!"

	# Instantiate multiple models.
	models = {"Multinomial Naive Bayes": MultinomialNB(), \
			  "Bernoulli Naive Bayes": BernoulliNB()}

	# For each model, fit it to the training data and make predictions, then evaluate on both sets for comparison purposes. We're interested in the test set scores.
	for model_name in models:
		model = models[model_name]
		predictions_train, predictions_test = fit_and_predict(model, \
											  train_features, y_train, test_features)

		print "Done fitting and predicting {0}!".format(model_name)

		score_train = evaluate(y_train, predictions_train)
		score_test = evaluate(y_test, predictions_test)
		print "{0} scores:\n{1} on train\n{2} on test set\n".format(model_name, score_train, score_test)

	elapsed = time.time() - start
	print "{0} seconds elapsed.".format(elapsed)

if __name__ == "__main__":
	main()