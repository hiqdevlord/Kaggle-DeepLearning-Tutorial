# Import a pre-trained model
from gensim.models import Word2Vec
model = Word2Vec.load("model_500features_50minwords_window10") # ChangeMe
word_vectors = model.syn0 # float32 array of vocab_size rows x num_features columns

#Word2Vec output is already normalized; shouldn't need to whiten the vectors
from sklearn.cluster import KMeans
import time

start = time.time() # Start timer

# 500 clusters, 500 features, ~15k vocab took ~11 minutes => 81% correct
num_clusters = 1500
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

end = time.time() 
elapsed = end - start
print "Time taken for sklearn clustering: ", elapsed, "seconds."

# Create a Word -> Index dictionary
word_centroid_map = dict(zip( model.index2word, idx ))

# Now need some code to convert paragraphs into bags of centroids
import pandas as pd
train = pd.read_csv("trainData.tsv", header=0,delimiter="\t",quoting=3)
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
def review_to_wordlist(review,remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text() 
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    review_text = re.sub(r'(.)\1+', r'\1\1',review_text) # replace doubled up letters
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)

clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append(review_to_wordlist(review,remove_stopwords=True))

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review,remove_stopwords=True))

def create_bag_of_centroids( wordlist, word_centroid_map ):
    n_centroids = max(word_centroid_map.values()) + 1
    bag_of_centroids = np.zeros(n_centroids,dtype="float32")
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    return bag_of_centroids

# Output of this should be a n_reviews x n_centroids array
train_centroids = np.zeros((train["review"].size, num_clusters),dtype="float32")
counter = 0.
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter = counter+1
    print counter


# Fit a simple classifier such as logreg or RF 
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
print "Fitting a random forest to labeled training data..."
forest = forest.fit(trainDataVecs,train["sentiment"])


# Convert test reviews to bags of centroids
test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")
counter = 0.
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter = counter + 1
    print counter

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

