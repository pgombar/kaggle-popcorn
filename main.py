from evaluation import evaluate_model_feature
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from features import *
#from sklearn.ensemble import RandomForestClassifier

# Set of classifier models
#		   key				model name				model
models = {"mnb" :	("Multinomial Naive Bayes",	MultinomialNB()), \
		  "bnb":	("Bernoulli Naive Bayes",	BernoulliNB())}

# Set of feature extractors
#				key				name				feature class
features = {"tftidf": ("Term-Frequency times Inverse Document-Frequency", Tf_Idf()), \
			"bow":	  ("Bag Of Words" ,				BoW() )}

def test_all(ratio = 0.7):
	for model_key in models:
		model = models[model_key]
		for feature_key in features:
			feature = features[feature_key]
			evaluate_model_feature(model[0], model[1], feature[0], feature[1], ratio)

def test(model_key, feature_key, ratio = 0.7):
	model = models[model_key]
	feature = features[feature_key]
	evaluate_model_feature(model[0], model[1], feature[0], feature[1], ratio)

def main():
	test('mnb','tftidf', 1)
	#test_all()
	
if __name__ == "__main__":
	main()