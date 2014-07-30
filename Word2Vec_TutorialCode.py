
# See https://code.google.com/p/word2vec/ and http://radimrehurek.com/gensim/models/word2vec.html
# Note that the former is in C and not well written. Required manual debugging to run on my machine.

# 1. Install gensim, which includes the Python implementation of word2vec
# 2. Install cython <- May be problematic for Windows users -- resulting in 70x slowdown
# 3. Download the data
#
# If installing skikit-learn: sudo port install py27-scikit-learn (or appropriate
#     python version if not 2.7 (uses MacPorts)

# This assumes you're already in the directory containing the data files

import logging
from gensim.models import word2vec
import numpy as np

import pandas as pd
from bs4 import BeautifulSoup
import re, string
from nltk.corpus import stopwords

# This controls word2vec output
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def review_to_words(review):
    review_text = BeautifulSoup(review).get_text() 
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    return(words)

def review_to_sentences(review):
    review_text = BeautifulSoup(review).get_text()
    raw_sentences = string.split(review_text,sep=".")
    sentences = []
    for raw_sentence in raw_sentences:
        sentence = re.sub("[^a-zA-Z]"," ",raw_sentence)
        if len(sentence) > 0:
            sentences.append(sentence.lower().split())
    return sentences

# Read data from files
train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",header=0,delimiter="\t",quoting=3)

print "Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled "\
    "reviews\n" % (train["review"].size, test["review"].size, unlabeled_train["review"].size )

num_features = 1024 # should be a multiple of 4 for optimal speed but can be anything. Lower -> faster
min_word_count = 10 # Set to at least some reasonable value like 10. Higher -> faster
num_workers = 4 # Number of threads to run in parallel. Varies by machine but at least 4 is a safe bet

# Can verify that the parallization is working by using >top -o cpu. The Python process should spin up to usage
# of around num_workers * 100%
# The num_workers parameter has NO EFFECT IF CYTHON IS NOT INSTALLED AND WORKING PROPERLY!!

# Train on the smaller set first -- to illustrate the differences in accuracy between 25k model and 75k model
# min_count affects vocabulary size and is the minimum times a word must appear to be included in the model
# size affects the number of features that each word vector will have (optimized if a multiple of 4)
# workers indicates cores to use for parallelization. Only takes effect if cython is installed

sentences = []
print "Parsing sentences from training set"
for review in train["review"]:
    sentences += review_to_sentences(review)

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review)

print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features,  min_count = min_word_count)

# The below makes the model more memory efficient but seals it off from further training
model.init_sims(replace=True)

# In the tutorial, also make a note that they can save / load this model - train it more later

# ************************************

def makeFeatureVec(words, model, num_features):
    # Utility function to create an average word vector for a given review
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # Convert index2word to a set, for speed
    index2word_set = set(model.index2word)    
    for word in words:
        if word in index2word_set:  # index2word returns the vocabulary list for the model
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate the average feature vector
    # and return a 2D numpy array
    counter = 0.
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        if counter%1000. == 0.:
            print "Review %d of %d" % (counter, len(reviews))
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter = counter + 1.
    return reviewFeatureVecs

# NOTE: The vector averaging is a bit slow (despite some minor optimizations such as matrix preallocation)
#
# Note that this operation is 'embarassingly parallel' and is a good candidate for multi-threading
# if the tutorial wants to go into that; could use the python package pp 
# http://www.parallelpython.com/content/view/15/30/#QUICKSMP

print "Creating average feature vectors for labeled reviews..."

# Unlike the first step, we now need to parse the reviews as a whole, not as individual sentences
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_words(review))

trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features)

# Fit a simple classifier such as logreg or RF 
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)

print "Fitting a random forest to labeled training data..."
forest = forest.fit(trainDataVecs,train["sentiment"])

print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_words(review))

# Slow; see comments above - good candidate for parallelizing if we want to go that route
testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )

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
# Results notes:  128 features x 50 min words + RF w/ 100 estimators -> 79.8% correct
# 128 features x 25 min words + RF w/ 100 estimators -> 79.2% correct
# 256 features x 25 min words + RF w/ 100 estimators -> 79.7% correct
# 256 features x 25 min words + RF w/ 200 estimators -> 80.2% correct
# 256 features x 100 min words + RF w/ 100 estimators -> 79.0% correct

# Could also try hooking up different classifiers, though that's not the main point of the tutorial



