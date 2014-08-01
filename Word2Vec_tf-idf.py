
# See https://code.google.com/p/word2vec/ and http://radimrehurek.com/gensim/models/word2vec.html
# Note that the former is in C and not well written. Required manual debugging to run on my machine.

# 1. Install gensim, which includes the Python implementation of word2vec
# 2. Install cython <- May be problematic for Windows users -- resulting in 70x slowdown
# 3. Download the data
#
# This script assumes you're already in the directory containing the data files

from gensim.models import word2vec
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re, string


# Read data from files
train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",header=0,delimiter="\t",quoting=3)

print "Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled "\
    "reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size )

num_features = 4096 # should be a multiple of 4 for optimal speed but can be anything. Lower -> faster

# Load a pre-trained model
model = Word2Vec.load("model.word2vec")


def clean_review(review,remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text() 
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    review_text = re.sub(r'(.)\1+', r'\1\1',review_text) # replace doubled up letters
    words = review_text.lower().split()
    return(words)

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(" ".join(clean_review(review)))

# The vectorizer expects a list, with each review as one string (not a list of lists)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer = "word",   # Don't create n-grams
                             tokenizer = None,    # Could also call our own tokenizer
                             preprocessor = None, # Since we did our own
                             stop_words = "english")   


# Get tf-idf weights as a dictionary and pass them to a vector weighting function
vectorizer.fit_transform(clean_train_reviews)  
idf_weights = vectorizer._tfidf.idf_
feature_dict = dict(zip(vectorizer.get_feature_names(), idf_weights))


# In the tutorial, also make a note that they can save / load this model - train it more later
#
# ************************************

def makeFeatureVec(words, model, num_features, feature_dict):
    # Utility function to create an average word vector for a given review
    reviewFeatureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # Convert index2word to a set, for speed
    index2word_set = set(model.index2word)    
    dict_key_set = set(feature_dict.keys())
    for word in words:
        if word in index2word_set:  # index2word returns the vocabulary list for the model
            if word in dict_key_set: 
                nwords = nwords + 1.
                weightedWordVec = np.multiply(model[word],feature_dict[word])
                reviewFeatureVec = np.add(reviewFeatureVec,weightedWordVec)
    featureVec = np.divide(reviewFeatureVec,nwords)
    return featureVec


def getWeightedFeatureVecs(reviews, model, num_features, feature_dict):
    # Given a set of reviews (each one a list of words), calculate the average feature vector
    # and return a 2D numpy array
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        if counter%1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features, feature_dict)
        counter = counter + 1.
    return reviewFeatureVecs


# NOTE: The vector averaging is a bit slow (despite some minor optimizations such as matrix preallocation)
#
# Note that this operation is 'embarassingly parallel' and is a good candidate for multi-threading
# if the tutorial wants to go into that; could use the python package pp 
# http://www.parallelpython.com/content/view/15/30/#QUICKSMP

# the function 'getWeightedFeatureVecs' requires individual words, not a whole review (input
# should be a list of lists)
#
print "Creating average feature vectors for labeled reviews..."

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(clean_review(review))

trainDataVecs = getWeightedFeatureVecs( clean_train_reviews, model, num_features, feature_dict )


# Fit a simple classifier such as logreg or RF 
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)

print "Fitting a random forest to labeled training data..."
forest = forest.fit(trainDataVecs,train["sentiment"])

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_words(review))


# Slowish; see comments above - good candidate for parallelizing if we want to go that route
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(clean_review(review))

testDataVecs = getWeightedFeatureVecs( clean_test_reviews, model, num_features, feature_dict )

# Test & extract results
result = forest.predict(testDataVecs)

# Write the test results
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("Word2Vec.csv")

# **********************

# Compare to known results (internal use only)
        
known_result = pd.read_csv("testData-TRUTH.csv",header=0)
percent_correct = sum(known_result["sentiment"]==output["sentiment"])/25000.
print "Fraction correct = %f" % percent_correct

# Maybe also output a confusion matrix here

# ***********************
#
# With 4096 features and stopword removal (supervised portion only): 81.2% correct



