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
import general_functions as general_f

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
number = '4'  # The necessary number of times a word occurs to be in vocab
ignore = ''  # Whether to include my "by-hand" feature selection or not


# The actual sentiments for the training data
sentiment_training = np.loadtxt('output/out' + ignore + '_classes_' + number + '.txt',
                                unpack=True)

# Read in the vocab list
with open('output/out' + ignore + '_vocab_' + number + '.txt', 'r') as f:
    vocab_training = [line for line in f]

# Read in the bag of words representation of the data
bagofwords_training = general_f.read_bagofwords_dat('output/out' + ignore + '_bag_of_words_' + number + '.csv')
#print bagofwords_training
#print bagofwords_training[0]
#


classifier = naivebayes.train(convert_to_format_that_naivebayes_can_read(vocab_training, bagofwords_training, sentiment_training))
# classifier.show_most_informative_features(15)

# Now read in the test data
sentiment_test = np.loadtxt('output/test' + ignore + '_classes_' + number + '.txt',
                            unpack=True)
bagofwords_test = general_f.read_bagofwords_dat('output/test' + ignore + '_bag_of_words_' + number + '.csv')

print 'min number of words for vocab: ' + str(number)
#print 'accuracy of training data:' + str(nltkutil.accuracy(classifier, convert_to_format_that_naivebayes_can_read(vocab_training, bagofwords_training, sentiment_training)))

print 'accuracy of test data: ' + str(nltkutil.accuracy(classifier, convert_to_format_that_naivebayes_can_read(vocab_training, bagofwords_test, sentiment_test)))
print 'other stuff of test data: '
# predict_test = classifier.classify(convert_to_format_that_naivebayes_can_read(vocab_training, bagofwords_test, sentiment_test))
# print general_f.precision_recall_etc(predict_test, sentiment_test)
classified_sentiments = []
for i in range(len(bagofwords_test)):
    classified_sentiments.append(classifier.classify(bagofwords_test[i]))

print general_f.precision_recall_etc(classified_sentiments, sentiment_test)
