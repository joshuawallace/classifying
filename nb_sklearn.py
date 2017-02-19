#############
# Created 2-18-17 by JJW
# HW1 COS 424
# Implements scikit-learn's Gaussian NB classifier
# on some reviews
#
# AGBTG
# ###########
# This webiste helped me: http://scikit-learn.org/stable/modules/naive_bayes.html


from sklearn.naive_bayes import GaussianNB
import numpy as np
import datareader as datareader


number = '6'
ignore = '_ignore'

# The actual sentiments for the training data
sentiment_training = np.loadtxt('output/out' + ignore + '_classes_' + number + '.txt',
                                unpack=True)

# Read in the bag of words representation of the data
bagofwords_training = datareader.read_bagofwords_dat('output/out' + ignore + '_bag_of_words_' + number + '.csv')

sentiment_test = np.loadtxt('output/test' + ignore + '_classes_' + number + '.txt',
                            unpack=True)
bagofwords_test = datareader.read_bagofwords_dat('output/test' + ignore + '_bag_of_words_' + number + '.csv')

classifier = GaussianNB().fit(bagofwords_training, sentiment_training)

predict_training = classifier.predict(bagofwords_training)
predict_test     = classifier.predict(bagofwords_test)