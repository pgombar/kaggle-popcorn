from cache import auto_cache
import pandas, numpy
_rnd_seed = 42
_split_ratio = 0.7

_labelled_path   = "data/labeledTrainData.tsv"	 # For supervised learning
_test_path 	     = "data/testData.tsv"			 # For Kaggle score
_unlabelled_path = "data/unlabeledTrainData.tsv" # For Unsupervised learning
_output_path	 = "output.csv"					 # Output
def _rprint(str): # Next print overwrites this, eg use for indicate progress
	sys.stdout.write("  Preprocessing : " + str + "            \r")
	sys.stdout.flush()

def _get_clean_data(path): 				# cleans raw data

	def _review_to_words(raw_review , stops): # cleans a review
		review_text = BeautifulSoup(raw_review, "html.parser").get_text().lower()
		letters_only = re.sub("[^a-z]", " ", review_text)
		words = letters_only.split()
		meaningful_words = [w for w in words if not w in stops]
		return( " ".join( meaningful_words ))
		
	_rprint("Loading raw train data...")
	data = pandas.read_csv(path, header=0, delimiter="\t", quoting=3)
	_rprint("Building stopwords dictionary...")
	clean_reviews, reviews = [], data["review"]
	stops = set(stopwords.words("english")) # precalculating makes it faster
	n, i = len(reviews), 0
	for rev in reviews:
		if i % (n/200) == 0:
			_rprint("Cleaning reviews (%d %%)" % 100*i/n)
		clean_reviews.append(_review_to_words(rev, stops))
		i = i + 1
	if "sentiment" in data:
		return clean_reviews, numpy.array(data["sentiment"])
	else:
		return clean_reviews,  numpy.array(data["id"])

def _get_split_data():
	clean_reviews, sentiment, ids = auto_cache(_get_clean_data, _labelled_path)
	_rprint("Splitting cleaned data...                         ")
	X_train, X_test, Y_train, Y_test = train_test_split(clean_reviews, sentiment, \
											train_size=_split_ratio, random_state=_rnd_seed)
	del data
	return X_train, X_test, Y_train, Y_test

def get_split_train_data(): # X, Y
	def gstraind():
		X_train, X_test, Y_train, Y_test = _get_split_data()
		return X_train, Y_train;
	return auto_cache(gstraind)

def get_split_test_data():  # X, Y
	def gstestd():
		X_train, X_test, Y_train, Y_test = _get_split_data()
		return X_test, Y_test;
	return auto_cache(gstestd)

def get_orig_train_data(): # X, Y
	return auto_cache(_get_clean_data,_labelled_path)
	
def get_orig_test_data(): # X, id
	return auto_cache(_get_clean_data,_test_path)

def get_unlabelled_data():
	X, id = auto_cache(_get_clean_data, _unlabelled_path)
	return X
	
def write_predictions(ids, predictions):
	output = pd.DataFrame( data={"id":ids, "sentiment":predictions} )
	output.to_csv(_output_path, index = False, quoting = 3)