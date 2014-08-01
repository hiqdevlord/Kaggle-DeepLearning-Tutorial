import numpy as np
import pandas as pd
import time

start = time.time() # Start timer

train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)

# > sudo pip install BeautifulSoup4
# In general, not a good idea to use regex to strip HTML
# Might be ok in this case though

print train["review"][1]
from bs4 import BeautifulSoup
example1 = BeautifulSoup(train["review"][1])
print example1.get_text()

# This has the tag \x85 - python decode method w/ UTF-8 doesn't get rid of it
# Using 'print' makes us not see it on screen though - good enough for me!

# Remove numbers & punctuation; tokenize
import re
example1_alpha = re.sub("[^a-zA-Z]"," ", example1.get_text())
words = example1_alpha.lower().split()

# Tokenize
#from nltk.tokenize import RegexpTokenizer
#tokenizer = RegexpTokenizer(r'\w+')
#words = tokenizer.tokenize(example1_alpha.lower()) # returns a list

# Remove stopwords
from nltk.corpus import stopwords
words = [w for w in words if not w in stopwords.words('english')]

# Note: got error like "Resource 'corpora/stopwords' not found. Please use the NLTK Downloader.."
# Include a note about this in the tutorial
# Do:
# import nltk
# nltk.download()
# Download all the corpora

# Porter stemming example
# from nltk.stem import PorterStemmer  # May or may not want to stem
# stemmer = PorterStemmer()
# filtered_words = [stemmer.stem(w) for w in words if not w in stopwords.words('english')]
# Also note that we could "lemmatize" (but aren't going to)

# Put it together into a function
def review_to_words(review):
    review_text = BeautifulSoup(review).get_text()     # Remove any HTML 
    review_text = re.sub("[^a-zA-Z]"," ", review_text) # Remove any numbers & punctuation
    review_text = re.sub(r'(.)\1+', r'\1\1',review_text) # replace doubled up letters
    words = review_text.lower().split()                # Convert to lower case, split into words
    stops = set(stopwords.words("english")) # Sets faster than lists!
    meaningful_words = [w for w in words if not w in stops]  # Remove stopwords
    return(" ".join(meaningful_words))  # Pass the words back to the caller as a single "document"   

num_reviews = train["review"].size
clean_train_reviews = []

print "Cleaning and parsing the training set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % (i+1,num_reviews)
    # NOTE: Because this is a list, using "append" vs preallocating makes almost no
    # difference in run time
    clean_train_reviews.append(review_to_words(train["review"][i]))

num_reviews = len(test.index)
clean_test_reviews = []

print "Cleaning and parsing the test set movie reviews...\n"
for i in xrange(0,num_reviews):
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % (i+1,num_reviews)
    clean_test_reviews.append(review_to_words(test["review"][i]))

# Print time taken so far
end = time.time() 
elapsed = end - start
print "Time taken to clean reviews: ", elapsed, "seconds."
start = time.time()

# Create bag of words from labeled train data (using sklearn)
# NOTE: Some difficulty (on home laptop) importing the feature extractor 
# - due to numpy / scipy disagreeing about sizes of integers?  
#
print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",  # Don't create n-grams
                             tokenizer = None, # Could also call our own tokenizer
                             preprocessor = None, # Since we did our own
                             stop_words = None, # Since we already removed them
                             max_features = 5000) # Max vocab

# "Fit" learns the vocabulary
# "Transform" returns the feature matrix
train_data_features = vectorizer.fit_transform(clean_train_reviews).toarray()
print train_data_features

vocab = vectorizer.get_feature_names()
print vocab

dist = np.sum(train_data_features, axis=0)
for tag, count in zip(vocab, dist):     
    print count, tag

# Feed it into a supervised learning algorithm
print "Training the random forest (this may take a while)...\n"
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data_features,train["sentiment"])

end = time.time() 
elapsed = end - start
print "Time taken to extract features and fit model: ", elapsed, "seconds."

# Get test results.  NOTE that we do NOT call "fit" for the test set
test_data_features = vectorizer.transform(clean_test_reviews).toarray() 
result = forest.predict(test_data_features)

# Write out test results
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("Bag_of_Words.csv")

# Compare to the known results here (internal use only)
known_result = pd.read_csv("testData-TRUTH.csv",header=0)
percent_correct = sum(known_result["sentiment"]==output["sentiment"])/25000.
print "Final fraction correct = %f\n" % (percent_correct,)

# Results
# 10,000 word vocab -> 84.9% correct
# 5,000 word vocab -> 84.5% correct

