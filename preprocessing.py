# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

data_path = "data/labeledTrainData.tsv"


def get_data(filename):
    return pd.read_csv(filename, header=0, delimiter="\t", quoting=3)


def review_to_words(raw_review):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #    a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    #    and return the result.
    return(" ".join( meaningful_words))


def clean_data(pd_data):
    # Initialize an empty list to hold the clean reviews
    clean_reviews = []
    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list
    num_reviews = pd_data["review"].size
    for i in xrange(0, num_reviews):
        clean_reviews.append(review_to_words(pd_data["review"][i]))

    return clean_reviews


def get_train_test_sets():
    # Reads data, cleans it and splits it into a train and test set.
    data = get_data(data_path)
    clean_reviews = clean_data(data)

    X_train, X_test, y_train, y_test = train_test_split(clean_reviews, np.array(data["sentiment"]), train_size=0.7, random_state=42)
    
    return X_train, X_test, y_train, y_test
