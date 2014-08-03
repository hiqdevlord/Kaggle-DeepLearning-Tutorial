# Import a pre-trained model
from gensim.models import Word2Vec
model = Word2Vec.load("300features_40min_word_count_10context") # ChangeMe
word_vectors = model.syn0 # float32 array of vocab_size rows x num_features columns

from sklearn.cluster import KMeans
import time

start = time.time() # Start timer

# 250 clusters, 500 features, ~15k vocab -> 81.7% correct 
# 500 clusters, 500 features, ~15k vocab took ~11 minutes  -> 82.81% correct
# 1000 clusters, 500 features, ~15k vocab took 22 minutes -> 82.9% normalized, 82.8% not normalized
# 1500 clusters, 500 features, ~15k vocab => 33 minutes -> 83.6% correct
# about 5000 clusters -> 3729 seconds -> 84.5% correct

# Basic conclusion: this is time consuming and doesn't work better than bag of words!

num_clusters = word_vectors.shape[0] / 5
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

end = time.time() 
elapsed = end - start
print "Time taken for sklearn clustering: ", elapsed, "seconds."

# Create a Word -> Index dictionary
word_centroid_map = dict(zip( model.index2word, idx ))

# Now need some code to convert paragraphs into bags of centroids
import pandas as pd
train = pd.read_csv("labeledtrainData.tsv", header=0,delimiter="\t",quoting=3)
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
    if (counter % 5000) == 0:
        print "review %d of %d" % (counter, len(clean_train_reviews))
    train_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter = counter+1


# Fit a simple classifier such as logreg or RF 
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
print "Fitting a random forest to labeled training data..."
forest = forest.fit(train_centroids,train["sentiment"])


# Convert test reviews to bags of centroids
test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")
counter = 0.
for review in clean_test_reviews:
    if (counter % 5000) == 0:
        print "review %d of %d" % (counter, len(clean_test_reviews))
    test_centroids[counter] = create_bag_of_centroids( review, word_centroid_map )
    counter = counter + 1

# Test & extract results
result = forest.predict(test_centroids)

# Write the test results
output = pd.DataFrame(data={"id":test["id"], "sentiment":result})
output.to_csv("BagOfCentroids.csv")


# **********************

# Compare to known results (internal use only)
        
known_result = pd.read_csv("testData-TRUTH.csv",header=0)
percent_correct = sum(known_result["sentiment"]==output["sentiment"])/25000.
print "Fraction correct = %f" % percent_correct

