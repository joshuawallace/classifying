####################
# Created by JJW Feb 18 2017
# 
# This gets used with the various other machine learning codes
# to read in the training and test data
#####################

import numpy as np


# This file reads in the csv output from preprocessSentences.py
# that is the bag of words.  This is borrowed from the isntructors'
# preprocessSentences.py script.  Thank you!
def read_bagofwords_dat(myfile):
    bagofwords = np.genfromtxt(myfile, delimiter=',')
    return bagofwords


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


# This calculates the the precision, recall, specificity, sensitivity, and f1
# Returns a dict
def precision_recall_etc(classified_sentiment, actual_sentiment):
    if len(classified_sentiment) != len(actual_sentiment):
        raise RuntimeError("Lengths of arguments to accuracy_percentage not the same")
    tp=fp=tn=fn = 0  # true/false postiive/negative
    for i in range(len(classified_sentiment)):
        if actual_sentiment[i] == 1:  # actual sentiment is positive
            if classified_sentiment[i] == actual_sentiment[i]:
                tp += 1
            else:
                fn += 1
        else:  # actual sentiment is negative
            if classified_sentiment[i] == actual_sentiment[i]:
                tn += 1
            else:
                fp += 1

    precision = float(tp)/float(tp + fp)
    recall = float(tp)/float(tp + fn)
    specificity = float(tn)/float(fp + tn)
    NPV = float(tn)/float(tn + fn)
    f1 = 2.*float(precision*recall)/float(precision + recall)

    return {'precision': precision, 'recall': recall,
            'specificity': specificity, 'NPV': NPV,
            'f1': f1}