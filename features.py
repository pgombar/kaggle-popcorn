from sklearn.feature_extraction.text import \
	TfidfVectorizer,	\
	CountVectorizer,	\
	FeatureHasher
import numpy as np
import nltk
# See more at: http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
from scipy.sparse import vstack, hstack

class Tf_Idf:
	name = "Term-Frequency times Inverse Document-Frequency"
	tfidf = 0
	def fit_transform(self, X_train):
		self.tfidf = TfidfVectorizer(max_features = 10000)
		return self.tfidf.fit_transform(X_train)

	def transform(self, X_test):
		return self.tfidf.transform(X_test)
		
class BoW:
	name = "Bag Of Words"
	vectorizer = 0
	def fit_transform(self, X_train):
		self.vectorizer = CountVectorizer(analyzer = "word",		\
							tokenizer = None, preprocessor = None,	\
							stop_words = None, max_features = 5000)
		return self.vectorizer.fit_transform(X_train)

	def transform(self, X_test):
		return self.vectorizer.transform(X_test)

class PosNeg:#Panni
    name = "Set of Positive and Negative words"
    pos, neg = 0, 0
    def fit_transform(self, X_set):
		self.pos, self.neg = self.wordsFromFiles()
		return self.transform(X_set)
    
    def transform(self, X_set):
        feature_vector_all = []
        for review in X_set:
            p, n = 0, 0
            for word in review.split():
                if word in self.pos:
                    p = p + 1
                if word in self.neg:
                    n = n + 1
            feature_vector_all.append([p,n, p-n])
        return feature_vector_all
    
    def wordsFromFiles(self):
        with open('positive-words.txt', 'r') as p:
            pos = p.read()
        p.close()
        with open('negative-words.txt', 'r') as n:
            neg = n.read()
        n.close()
        return set(pos.split()), set(neg.split())

class Fconcat: # Concatenates features
	name = "Concatenated "
	features = [];
	def __init__(self, *Features):
		self.name, self.features = "Concatenated(", Features
		for f in self.features:
			self.name = self.name + f.name + " AND "
		self.name = self.name[0:len(self.name)-5] + ")"
	
	def fit_transform(self, X_train):
		ls = map(lambda f: f.fit_transform(X_train), self.features)
		return hstack(ls)
		#return np.concatenate(tuple(ls), axis = 1)

	def transform(self, X_test):
		ls = map(lambda f: f.transform(X_test), self.features)
		return hstack(ls)
		#return np.concatenate(tuple(ls), axis = 1)

class Fcustom:	# Same as Paula's custom_features in earlier versions
	name = "Word Count and Mean Word Length"
	def fit_transform(self, X_train):
		return self.transform(X_train)
		
	def transform(self, X_test):
		train_fs = []
		for review in X_test:
			train_fs.append(self.custom_feature_list(review))
		return np.asarray(train_fs);
	def custom_feature_list(self, review):
		words = review.split(' ')
		length = sum([len(word) for word in words])
		return [len(words), float(length) / len(words)]
