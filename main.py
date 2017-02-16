#############
# Created 2-14-17 by JJW
# This will be the main file for assignment 1 in COS 424
# For this assignment, I am performing a sentiment analysis
# on some reviews
#
# AGBTG
# ###########


import nltk.classify.util as nltkutil
from nltk.classify import NaiveBayesClassifier as naivebayes
import numpy as np


# This file reads in the csv output from preprocessSentences.py
# that is the bag of words.  This is borrowed from the isntructors'
# preprocessSentences.py script.  Thank you!
def read_bagofwords_dat(myfile):
    bagofwords = np.genfromtxt(myfile, delimiter=',')
    return bagofwords


# This converts the bag of words representation of each review
# into a dictionary format that can be read by the NaiveBayes
# function, and then converts that to the list of tuples (dict, sentiment)
# that NaiveBayes wants for input.
# Thanks to http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
# for guidance on how to make something like this.
def convert_to_format_that_naivebayes_can_read(vocabulary, bagofwords,
                                               sentiment):
    list_to_output = []
    for i in range(len(bagofwords)):
        features_dict = {}
        for j in range(len(vocabulary)):
            if bagofwords[i][j] > 0:
                features_dict[vocabulary[j]] = True
            else:
                features_dict[vocabulary[j]] = False
        list_to_output.append((features_dict, sentiment[i]))
    return list_to_output


# The number of words needing to appear in training set to be included
# as vocab.
number = '6'


# The actual sentiments for the training data
sentiment_training = np.loadtxt('output/out_classes_' + number + '.txt',
                                unpack=True)

# The 'class' of the training data, literally just the number line it is in
# original file
class_training     = np.loadtxt('output/out_samples_class_' + number + '.txt',
                                unpack=True)

# Read in the vocab list
with open('output/out_vocab_' + number + '.txt', 'r') as f:
    vocab_training = [line for line in f]

# Read in the bag of words representation of the data
bagofwords_training = read_bagofwords_dat('output/out_bag_of_words_' + number + '.csv')
#print bagofwords_training
#print bagofwords_training[0]
#


classifier = naivebayes.train(convert_to_format_that_naivebayes_can_read(vocab_training, bagofwords_training, sentiment_training))
# classifier.show_most_informative_features(15)

# Now read in the test data
sentiment_test = np.loadtxt('output/test_classes_' + number + '.txt',
                            unpack=True)
class_test     = np.loadtxt('output/test_samples_class_' + number + '.txt',
                            unpack=True)
bagofwords_test = read_bagofwords_dat('output/test_bag_of_words_' + number + '.csv')

print 'accuracy of test data:' + str(nltkutil.accuracy(classifier, convert_to_format_that_naivebayes_can_read(vocab_training, bagofwords_test, sentiment_test)))
