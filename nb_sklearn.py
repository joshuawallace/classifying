#############
# Created 2-18-17 by JJW
# HW1 COS 424
# Implements scikit-learn's NB classifier
# on some reviews, uses all three of the Gaussian,
# Bernoulli, and multinomial estimators
#
# AGBTG
# ###########
# This website helped me: http://scikit-learn.org/stable/modules/naive_bayes.html


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import general_functions as general_f


numbers_array = [3, 4, 5, 6]  # The different thresholds to use to count vocab
ignore = ''  # '_ignore'  # Whether to include my "by-hand" feature selection or not

for num in numbers_array:  # For the different threshold values
    number = str(num)  # The necessary number of times a word occurs to be in vocab

    # Print out the number of words being used as vocabulary
    print "############################"
    print "Number of words to be included as vocab is: " + number
    print ""

    # The actual sentiments for the training data
    sentiment_training = np.loadtxt('output/out' + ignore + '_classes_' + number + '.txt',
                                    unpack=True)

    # Read in the bag of words representation of the data
    bagofwords_training = general_f.read_bagofwords_dat('output/out' + ignore + '_bag_of_words_' + number + '.csv')

    # The same for the test data
    sentiment_test = np.loadtxt('output/test' + ignore + '_classes_' + number + '.txt',
                                unpack=True)
    bagofwords_test = general_f.read_bagofwords_dat('output/test' + ignore + '_bag_of_words_' + number + '.csv')

    # First, the Gaussian NB classifier
    classifier = GaussianNB().fit(bagofwords_training, sentiment_training)

    # Predict sentiment of the test data
    predict_test     = classifier.predict(bagofwords_test)

    # Calculate performance and print
    test_percentage     = general_f.accuracy_percentage(predict_test, sentiment_test)
    print "Gauss NB Test percentage for " + number + " :     " + str(test_percentage)
    print 'other stuff of test data: '
    print general_f.precision_recall_etc(predict_test, sentiment_test)

    # Now, for the Bernoulli NB classifier
    classifier = BernoulliNB().fit(bagofwords_training, sentiment_training)

    # Predict sentiment of the test data
    predict_test     = classifier.predict(bagofwords_test)

    # Calculate accuracy and print
    print ""
    test_percentage     = general_f.accuracy_percentage(predict_test, sentiment_test)
    print "BernNB Test percentage for " + number + " :     " + str(test_percentage)
    print 'other stuff of test data: '
    print general_f.precision_recall_etc(predict_test, sentiment_test)

    # And finally, for the Multinomial NB classifier
    classifier = MultinomialNB().fit(bagofwords_training, sentiment_training)

    # Predict sentiment of the test data
    predict_test     = classifier.predict(bagofwords_test)

    # Calculate accuracy and print
    print ""
    test_percentage     = general_f.accuracy_percentage(predict_test, sentiment_test)
    print "MultNB Test percentage for " + number + " :     " + str(test_percentage)
    print 'other stuff of test data: '
    print general_f.precision_recall_etc(predict_test, sentiment_test)
