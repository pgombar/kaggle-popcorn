from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
# See more at: http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting

class Tf_Idf:
	name = "Term-Frequency times Inverse Document-Frequency"
	tfidf = 0
	def fit_transform(self, X_train):
		self.tfidf = TfidfVectorizer(max_features = 10000)
		return self.tfidf.fit_transform(X_train).toarray()

	def transform(self, X_test):
		return self.tfidf.transform(X_test).toarray()
		
class BoW:
	name = "Bag Of Words"
	vectorizer = 0
	def fit_transform(self, X_train):
		self.vectorizer = CountVectorizer(analyzer = "word",		\
							tokenizer = None, preprocessor = None,	\
							stop_words = None, max_features = 5000)
		return self.vectorizer.fit_transform(X_train).toarray()

	def transform(self, X_test):
		return self.vectorizer.transform(X_test).toarray()

class Fconcat:
	name = "Concatenated "
	feature1 = 0;	feature2 = 0;
	def __init__(self, Feature1, Feature2):
		self.feature1 = Feature1; self.feature2 = Feature2;
		self.name = "Concatenated(" + self.feature1.name \
						   + " AND " + self.feature2.name + ")";
	
	def fit_transform(self, X_train):
		train_fs1 = self.feature1.fit_transform(X_train)
		train_fs2 = self.feature2.fit_transform(X_train)
		return np.concatenate((train_fs1, train_fs2), axis = 1)

	def transform(self, X_test):
		train_fs1 = self.feature1.transform(X_test)
		train_fs2 = self.feature2.transform(X_test)
		return np.concatenate((train_fs1, train_fs2), axis = 1)

def custom_feature_list(review):
	words = review.split(' ')
	length = sum([len(word) for word in words])
	return [len(words), float(length) / len(words)]

class Fcustom:	# Same as Paula's custom_features
	name = "word count and mean word length"
	def fit_transform(self, X_train):
		return self.transform(X_train)
		
	def transform(self, X_test):
		train_fs = []
		for review in X_test:
			train_fs.append(custom_feature_list(review))
		return train_fs;

# Paula's original code:			

def custom_features(X_train, X_test): # TODO:revrite
	# Constructs feature vector of custom features.
	# Write your own function and append it to the feature vector.
	
	feature_vector_all_train = []
	feature_vector_all_test = []
	
	for (train_text, test_text) in zip(X_train, X_test):
		feature_vector_train = []
		feature_vector_train.append(text_length(train_text))
		feature_vector_train.append(average_word_length(train_text))
		
		feature_vector_test = []
		feature_vector_test.append(text_length(test_text))
		feature_vector_test.append(average_word_length(test_text))
		# Append other custom features here...
		
		# Append to master feature vector list
		feature_vector_all_train.append(feature_vector_train)
		feature_vector_all_test.append(feature_vector_test)
		
	return feature_vector_train, feature_vector_test


def text_length(raw_text):
    return len(raw_text.split(" "))


def average_word_length(raw_text):
    words = raw_text.split(" ")
    length = sum([len(word) for word in words])
    return float(length) / len(words)