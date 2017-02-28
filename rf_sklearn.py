#####################
# Created 2-23-17 by JJW
# HW1 COS 424
# Implements scikit-learn's Random Forest classifier
# on some reviews.
#
# AGBTG
# ###########
# This website helped:
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html


import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFC
import general_functions as general_f

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

# Define and train the classifier
classifier = RFC(n_estimators=150, criterion='entropy', n_jobs=3, verbose=0).fit(bagofwords_training, sentiment_training)

# Predict sentiment of the test data
predict_test     = classifier.predict(bagofwords_test)

# Calculate accuracy and print
print ""
test_percentage     = general_f.accuracy_percentage(predict_test, sentiment_test)
print "RFC Test percentage for " + number + " :     " + str(test_percentage)
print 'other stuff of test data: '
print general_f.precision_recall_etc(predict_test, sentiment_test)
