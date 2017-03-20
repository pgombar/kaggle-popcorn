from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def tf_idf(X_train, y_train, X_test):
    # Constructs a feature vector of tf-idf scores.
    # See more at: http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting

    tfidf = TfidfVectorizer()
    train_features = tfidf.fit_transform(X_train).toarray()
    test_features = tfidf.transform(X_test).toarray()

    return train_features, test_features


def bow(X_train, y_train, X_test):
    # Constructs a bag of words feature vector.
    # See more at: http://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation

    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000)
    train_features = vectorizer.fit_transform(X_train).toarray()
    test_features = vectorizer.transform(X_test).toarray()

    return train_features, test_features


def custom_features(X_train, y_train, X_test):
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