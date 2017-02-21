#############
# Created 2-18-17 by JJW
# HW1 COS 424
# Implements scikit-learn's Gaussian NB classifier
# on some reviews
#
# AGBTG
# ###########
# This website helped me: http://scikit-learn.org/stable/modules/naive_bayes.html


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import datareader as datareader


# This evaluates the percentage of classified comments that were
# correctly classified
def accuracy_percentage(classified_sentiment, actual_sentiment):
    if len(classified_sentiment) != len(actual_sentiment):
        raise RuntimeError("Lengths of arguments to accuracy_percentage not the same")        
    number_correct = 0
    for i in range(len(classified_sentiment)):
        if classified_sentiment[i] == actual_sentiment[i]:
            number_correct += 1

    return float(number_correct)/float(len(actual_sentiment))


numbers_array = [3, 4, 5, 6]
ignore = ''

for num in numbers_array:
    number = str(num)
    print "############################"
    print "Number of words to be included as vocab is: " + number
    print ""

    # The actual sentiments for the training data
    sentiment_training = np.loadtxt('output/out' + ignore + '_classes_' + number + '.txt',
                                    unpack=True)

    # Read in the bag of words representation of the data
    bagofwords_training = datareader.read_bagofwords_dat('output/out' + ignore + '_bag_of_words_' + number + '.csv')

    sentiment_test = np.loadtxt('output/test' + ignore + '_classes_' + number + '.txt',
                                unpack=True)
    bagofwords_test = datareader.read_bagofwords_dat('output/test' + ignore + '_bag_of_words_' + number + '.csv')

    # First, the Gaussian NB classifier
    classifier = GaussianNB().fit(bagofwords_training, sentiment_training)

    predict_training = classifier.predict(bagofwords_training)
    predict_test     = classifier.predict(bagofwords_test)

    # Calculate accuracy and print
    training_percentage = accuracy_percentage(predict_training, sentiment_training)
    print "GaussNB Training percentage for " + number + " : " + str(training_percentage)
    test_percentage     = accuracy_percentage(predict_test, sentiment_test)
    print "Gauss NB Test percentage for " + number + " :     " + str(test_percentage)

    # Now, for the Bernoulli NB classifier
    classifier = BernoulliNB().fit(bagofwords_training, sentiment_training)

    predict_training = classifier.predict(bagofwords_training)
    predict_test     = classifier.predict(bagofwords_test)

    # Calculate accuracy and print
    print ""
    training_percentage = accuracy_percentage(predict_training, sentiment_training)
    print "BernNB Training percentage for " + number + " : " + str(training_percentage)
    test_percentage     = accuracy_percentage(predict_test, sentiment_test)
    print "BernNB Test percentage for " + number + " :     " + str(test_percentage)

    # And finally, for the Multinomial NB classifier
    classifier = MultinomialNB().fit(bagofwords_training, sentiment_training)

    predict_training = classifier.predict(bagofwords_training)
    predict_test     = classifier.predict(bagofwords_test)

    # Calculate accuracy and print
    print ""
    training_percentage = accuracy_percentage(predict_training, sentiment_training)
    print "MultNB Training percentage for " + number + " : " + str(training_percentage)
    test_percentage     = accuracy_percentage(predict_test, sentiment_test)
    print "MultNB Test percentage for " + number + " :     " + str(test_percentage)
