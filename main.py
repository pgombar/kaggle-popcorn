from evaluation import evaluate_model_feature
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from features import *
#from sklearn.ensemble import RandomForestClassifier

# Set of classifier models
#		   key				model name				model
models = {"mnb" :	("Multinomial Naive Bayes",	MultinomialNB()), \
		  "bnb":	("Bernoulli Naive Bayes",	BernoulliNB()),   \
		  "forest": ("Random Forest Classifier", RandomForestClassifier(n_estimators = 100)) }

# Set of feature extractors
#			 key	  feature class
features = {"tftidf": Tf_Idf(), 	\
			"bow":	  BoW() ,		\
			"custom": Fcustom(),	\
			"concat": Fconcat(BoW(), Tf_Idf(), Fcustom())}

def test_all(ratio = 0.7):
	for model_key in models:
		model = models[model_key]
		for feature_key in features:
			feature = features[feature_key]
			evaluate_model_feature(model[0], model[1], feature, ratio)

def test(model_key, feature_key, ratio = 0.7):
	model = models[model_key]
	feature = features[feature_key]
	evaluate_model_feature(model[0], model[1], feature, ratio)

def main():
	test('bnb','custom', 0.7)
	test('bnb','concat', 0.7)
	#test_all()
	
if __name__ == "__main__":
	main()