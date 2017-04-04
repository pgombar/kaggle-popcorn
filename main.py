from evaluation import evaluate_model_feature
from features import *
from models import *

# Set of classifier models
#		   key		  model
models = {"mnb" :	Mnb(),		\
		  "bnb":	Bnb(),		\
		  "forest": RForest(),	\
		  "superv": SemiSupervised(Mnb())}

# Set of feature extractors
#			 key	  feature class
features = {"tftidf": Tf_Idf(), 	\
			"bow"	: BoW() ,		\
			"custom": Fcustom(),	\
			"posneg": PosNeg(),		\
			"ngram" : Ngram(),		\
			"concat": Fconcat(Tf_Idf(), Ngram(), PosNeg())}

def main():
	try:
		test(Fconcat(Tf_Idf(), Ngram(), PosNeg()), RForest())
		test(Fconcat(Ngram(), PosNeg()), Mnb())
		test(Fconcat(Ngram(), Tf_Idf()), Mnb())
		test(Ngram(), Mnb())

		#test_all()
	
	except:
		print "\n"
		raise

def test_all(ratio = 0.7):
	for model_key in models:
		model = models[model_key]
		for feature_key in features:
			test(models[model_key], features[feature_key], ratio)

def test(feature, model, ratio = 0.7):
	evaluate_model_feature(model, feature, ratio)
		
if __name__ == "__main__":
	main()
