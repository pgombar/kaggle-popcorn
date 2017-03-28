# 1. Make sure yu have a 'cache' folder
# 2. Use only the get_train_data(ratio) or the get_test_data(ratio) function
#		after first call, they load cached data
#		ratio parameter defines the splitting ratio of the train data
# 3. For loading the final Test data, use Get_Testing_Data() function
#		Use the Write_Predictions(predictions) to write the data
#		into the format that Kaggle asks for
# 4. You can delelte the content of the cache folder to start over, or
# 		call preprocess(True), so it overwrites the existing cache
import pandas as pd
import numpy as np
from nltk.corpus				import stopwords
from bs4						import BeautifulSoup
from sklearn.model_selection	import train_test_split
import nltk, re, cPickle, os.path, sys, gzip

labelled_data_path = "data/labeledTrainData.tsv" # data path
Test_data_path = "data/testData.tsv" # data path
X_Train_cache_path = "cache/X_train_%.1f.bin"
Y_Train_cache_path = "cache/Y_train_%.1f.bin"
X_Test_cache_path = "cache/X_test_%.1f.bin"
Y_Test_cache_path = "cache/Y_test_%.1f.bin"
Test_cache_path = "cache/Test.bin"

def get_train_data(ratio = 0.7):	# Loads train data from cache (creates it if needed)
	preprocess(ratio)
	X_train, Y_train = load_train_data()
	#print "Preprocessing : Completed.                       "
	return X_train, Y_train
	
def get_test_data(ratio = 0.7):	# Loads test data from cache (creates it if needed)
	preprocess(ratio)
	X_test, Y_test = load_test_data()
	#print "Preprocessing : Completed.                       "
	return X_test, Y_test
	
def Get_Testing_Data():
	if not os.path.isfile(Test_cache_path):
		rprint("Loading raw test data")
		data = pd.read_csv(Test_data_path, header=0, delimiter="\t", quoting=3)
		pair = (clean_data(data), data['id'])
		del data
		rprint("Saving Test data to cache")
		save_zipped_pickle(pair, Test_cache_path)
	else:
		rprint("Loading Test data from cache")
		pair = load_zipped_pickle(Test_cache_path)
	return pair[0], pair[1]	# text, ids

def Write_Predictions(path, ids, predictions):
	output = pd.DataFrame( data={"id":ids, "sentiment":predictions} )
	output.to_csv(path, index = False, quoting = 3)
	
def preprocess(ratio = 0.7, rebuild_cache = False):
# Ratio*10 should be an integer, it is the ratio of train data
# If there is no cached data for the given ratio, it creates it
	exists = os.path.isfile(X_Train_cache_path % ratio)	\
		 and os.path.isfile(Y_Train_cache_path % ratio)	\
		 and os.path.isfile(X_Test_cache_path  % ratio)	\
		 and os.path.isfile(Y_Test_cache_path  % ratio)
	if not exists or rebuild_cache:
		X_train, X_test, Y_train, Y_test = get_train_test_sets(ratio)
		save_train_data(X_train, Y_train, ratio)
		save_test_data(X_test,  Y_test, ratio)

def rprint(str): # Next print overwrites this, eg use for indicate progress
	sys.stdout.write("Preprocessing : " + str + "            \r")
	sys.stdout.flush()

def review_to_words(raw_review , stops): # cleans a review
	review_text = BeautifulSoup(raw_review, "html5lib").get_text().lower()
	letters_only = re.sub("[^a-z]", " ", review_text)
	words = letters_only.split()
	meaningful_words = [w for w in words if not w in stops]
	return( " ".join( meaningful_words ))

def clean_data(pd_data): 				# cleans raw data
	rprint("Building stopwords dictionary...")
	clean_reviews = []
	stops = set(stopwords.words("english")) # precalculating makes it faster
	reviews = pd_data["review"];
	n = len(reviews); 	i = 0;
	for rev in reviews:
		if i % (n/200) == 0:
			rprint("Cleaning reviews (%d %%)" % (100*i/n))
		clean_reviews.append(review_to_words(rev, stops))
		i = i + 1
	return clean_reviews
	
def get_train_test_sets(ratio = 0.7):
	rprint("Loading raw train data...")
	data = pd.read_csv(labelled_data_path, header=0, delimiter="\t", quoting=3)
	clean_reviews = clean_data(data)
	rprint("Splitting cleaned data...")
	X_train, X_test, Y_train, Y_test = train_test_split(clean_reviews, np.array(data["sentiment"]), \
											train_size=ratio, random_state=42)
	del data
	return X_train, X_test, Y_train, Y_test
		
# serializes data and saves in a (somewhat) compressed format
def save_zipped_pickle(obj, filename, protocol=-1):
	with gzip.open(filename, 'wb') as f:
		cPickle.dump(obj, f, protocol)
		f.close()

# loads compressed data and restores original state
def load_zipped_pickle(filename):	# loads and unpacks
	with gzip.open(filename, 'rb') as f:
		loaded_object = cPickle.load(f)
		f.close()
		return loaded_object

def save_train_data(X_train, Y_train, ratio = 0.7):
	rprint("Saving train data to cache    (1  %)")
	save_zipped_pickle(X_train, X_Train_cache_path % ratio)
	rprint("Saving train data to cache    (80 %)")
	save_zipped_pickle(Y_train, Y_Train_cache_path % ratio)
	rprint("Saving train data to cache    (100%)")

def save_test_data(X_test, Y_test, ratio = 0.7):
	rprint("Saving test data to cache     (1  %)")
	save_zipped_pickle(X_test, X_Test_cache_path % ratio)
	rprint("Saving test data to cache     (80 %)")
	save_zipped_pickle(Y_test, Y_Test_cache_path % ratio)
	rprint("Saving test data to cache     (100%)")
	
def load_train_data(ratio = 0.7):
	rprint("Loading train data from cache (1 %)")
	X_train = load_zipped_pickle(X_Train_cache_path % ratio)
	rprint("Loading train data from cache (80 %)")
	Y_train = load_zipped_pickle(Y_Train_cache_path % ratio)
	rprint("Loading train data from cache (100%)")
	return X_train, Y_train

def load_test_data(ratio = 0.7):
	rprint("Loading test data from cache  (1 %)")
	X_test = load_zipped_pickle(X_Test_cache_path % ratio)
	rprint("Loading test data from cache  (80 %)")
	Y_test = load_zipped_pickle(Y_Test_cache_path % ratio)
	rprint("Loading test data from cache  (100%)")
	return X_test, Y_test
	