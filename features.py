from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# See more at: http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting

class Tf_Idf:
	tfidf = TfidfVectorizer()
	def fit_transform(self, X_train):
		self.tfidf = TfidfVectorizer(max_features = 20000)
		train_features = self.tfidf.fit_transform(X_train).toarray()
		return train_features;

	def transform(self, X_test):
		return self.tfidf.transform(X_test)
		
class BoW:
	vectorizer = CountVectorizer()
	def fit_transform(self, X_train):
		self.vectorizer = CountVectorizer(analyzer = "word",		\
							tokenizer = None, preprocessor = None,	\
							stop_words = None, max_features = 5000)
		train_features = self.vectorizer.fit_transform(X_train).toarray()
		return train_features;

	def transform(self, X_test):
		return self.vectorizer.transform(X_test)

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
    return [len(raw_text.split(" "))]


def average_word_length(raw_text):
    words = raw_text.split(" ")
    length = sum([len(word) for word in words])
    return float(length) / len(words)