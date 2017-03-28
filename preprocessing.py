# 1. Make sure yu have a 'cache' folder
# 2. Use only the get_train_data() or the get_test_data() function
#		after first call, they load cached data
# 3. Input folder: "data/labeledTrainData.tsv"
# 4. You can delelte the content of the cache folder to start over, or
# 		call preprocess(True), so it overwrites the existing cache
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import cPickle
import os.path
import sys
import gzip

data_path = "data/labeledTrainData.tsv" # data path
picle_dumb_protocol = -1;

def get_train_data():	# Loads train data from cache (creates it if needed)
	preprocess()
	X_train, Y_train = load_train_data()
	#print "Preprocessing : Completed.                       "
	return X_train, Y_train
	
def get_test_data():	# Loads test data from cache (creates it if needed)
	preprocess()
	X_test, Y_test = load_test_data()
	#print "Preprocessing : Completed.                       "
	return X_test, Y_test

def preprocess(rebuild_cache = False):	# If there is no cached data, this creates it
	exists = os.path.isfile('cache/X_train.bin') and os.path.isfile('cache/X_test.bin')	 \
		 and os.path.isfile('cache/Y_train.bin') and os.path.isfile('cache/Y_test.bin')
	if not exists or rebuild_cache:
		X_train, X_test, Y_train, Y_test = get_train_test_sets()
		save_train_data(X_train, Y_train)
		save_test_data (X_test,  Y_test)

def rprint(str): # Next print overwrites this, eg use for indicate progress
	sys.stdout.write("Preprocessing : " + str + "            \r")
	sys.stdout.flush()

def review_to_words(raw_review , stops): # cleans a review
	review_text = BeautifulSoup(raw_review, "html5lib").get_text().lower()
	letters_only = re.sub("[^a-z]", " ", review_text)
	words = letters_only.split()
	meaningful_words = [w for w in words if not w in stops]
	return( " ".join( meaningful_words ))

def clean_data(pd_data): # cleans raw data
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
	
def get_train_test_sets():
	rprint("Loading raw data...")
	data = pd.read_csv(data_path, header=0, delimiter="\t", quoting=3)
	clean_reviews = clean_data(data)
	rprint("Splitting cleaned data...")
	X_train, X_test, Y_train, Y_test = train_test_split(clean_reviews, np.array(data["sentiment"]), \
											train_size=0.7, random_state=42)
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

def save_test_data(X_test, Y_test):
	rprint("Saving test data to cache     (1  %)")
	save_zipped_pickle(X_test, 'cache/X_test.bin')
	rprint("Saving test data to cache     (80 %)")
	save_zipped_pickle(Y_test, 'cache/Y_test.bin')
	rprint("Saving test data to cache     (100%)")
	
def save_train_data(X_train, Y_train):
	rprint("Saving train data to cache    (1  %)")
	save_zipped_pickle(X_train, 'cache/X_train.bin')
	rprint("Saving train data to cache    (80 %)")
	save_zipped_pickle(Y_train, 'cache/Y_train.bin')
	rprint("Saving train data to cache    (100%)")

def load_test_data():
	rprint("Loading test data from cache  (1 %)")
	X_test = load_zipped_pickle('cache/X_test.bin')
	rprint("Loading test data from cache  (80 %)")
	Y_test = load_zipped_pickle('cache/Y_test.bin')
	rprint("Loading test data from cache  (100%)")
	return X_test, Y_test
	
def load_train_data():
	rprint("Loading train data from cache (1 %)")
	X_train = load_zipped_pickle('cache/X_train.bin')
	rprint("Loading train data from cache (80 %)")
	Y_train = load_zipped_pickle('cache/Y_train.bin')
	rprint("Loading train data from cache (100%)")
	return X_train, Y_train