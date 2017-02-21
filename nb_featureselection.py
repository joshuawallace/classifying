#############
# Created 2-21-17 by JJW
# HW1 COS 424
# Implements scikit-learn's Binomial NB classifier
# on some reviews, and then performs some feature selection,
# also using scikit-learn
#
# AGBTG
# ###########
# http://scikit-learn.org/stable/modules/feature_selection.html


from sklearn.naive_bayes import BernoulliNB
import numpy as np
import general_functions as general_f
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline


number = '3'  # The necessary number of times a word occurs to be in vocab
ignore = ''  # '_ignore'  # Whether to include my "by-hand" feature selection or not

# The actual sentiments for the training data
sentiment_training = np.loadtxt('output/out' + ignore + '_classes_' + number + '.txt',
                                unpack=True)

# Read in the bag of words representation of the data
bagofwords_training = general_f.read_bagofwords_dat('output/out' + ignore + '_bag_of_words_' + number + '.csv')

# The same for the test data
sentiment_test = np.loadtxt('output/test' + ignore + '_classes_' + number + '.txt',
                            unpack=True)
bagofwords_test = general_f.read_bagofwords_dat('output/test' + ignore + '_bag_of_words_' + number + '.csv')

# Variance threshold
thresholds_to_try = np.linspace(0, 0.05, 10)
print "Value to compare against: " + str(len(bagofwords_training[0]))
for val in thresholds_to_try:
    selector = VarianceThreshold(threshold=val)
    print selector.fit_transform(bagofwords_training, sentiment_training)
    # print selector.variances_


# Now, the Bernoulli NB classifier
# classifier = BernoulliNB().fit(bagofwords_training, sentiment_training)

