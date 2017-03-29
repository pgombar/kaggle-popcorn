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
	features = [];
	def __init__(self, *Features):
		self.features = Features
		self.name = "Concatenated("
		for f in self.features:
			self.name = self.name + f.name + " and "
		self.name = self.name[0:len(self.name)-5] + ")"
	
	def fit_transform(self, X_train):
		ls = map(lambda f: f.fit_transform(X_train), self.features)
		return np.concatenate(tuple(ls), axis = 1)

	def transform(self, X_test):
		ls = map(lambda f: f.transform(X_test), self.features)
		return np.concatenate(tuple(ls), axis = 1)

def custom_feature_list(review):
	words = review.split(' ')
	length = sum([len(word) for word in words])
	return [len(words), float(length) / len(words)]

class Fcustom:	# Same as Paula's custom_features in earlier versions
	name = "word count and mean word length"
	def fit_transform(self, X_train):
		return self.transform(X_train)
		
	def transform(self, X_test):
		train_fs = []
		for review in X_test:
			train_fs.append(custom_feature_list(review))
		return np.asarray(train_fs);