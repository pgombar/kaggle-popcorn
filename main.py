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
			"concat": Fconcat(Tf_Idf(), BoW(), Fcustom(), PosNeg())}

def main():
	try:
	
		test('superv','custom', 0.7)
		#test('bnb','hash', 0.7)
		#test('bnb','custom', 0.7)
		#test_all()
	
	except:
		print "\n"
		raise

def test_all(ratio = 0.7):
	for model_key in models:
		model = models[model_key]
		for feature_key in features:
			feature = features[feature_key]
			evaluate_model_feature(model, feature, ratio)

def test(model_key, feature_key, ratio = 0.7):
	model, feature = models[model_key], features[feature_key]
	evaluate_model_feature(model, feature, ratio)
		
if __name__ == "__main__":
	main()
