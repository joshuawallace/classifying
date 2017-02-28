####################
# Created by JJW Feb 18 2017
#
# This file contains functions used with the various other codes
# in this directory to read in the training and test data.  It also
# contains functions to calculate the performance of the classifiers
#####################

import numpy as np


# This file reads in the csv output from preprocessSentences.py
# that is the bag of words.  This is borrowed from the instructors'
# preprocessSentences.py script.  Thank you!
# Returns a numpy array derived from a csv file
def read_bagofwords_dat(myfile):
    bagofwords = np.genfromtxt(myfile, delimiter=',')
    return bagofwords


# This evaluates the percentage of classified comments that were
# correctly classified
# Returns a number (float)
def accuracy_percentage(classified_sentiment, actual_sentiment):
    if len(classified_sentiment) != len(actual_sentiment):  # if lengths don't match
        raise RuntimeError("Lengths of arguments to accuracy_percentage not the same")
    number_correct = 0
    for i in range(len(classified_sentiment)):
        if classified_sentiment[i] == actual_sentiment[i]:  # if the predicted sentiment matches actual
            number_correct += 1

    return float(number_correct)/float(len(actual_sentiment))


# This calculates the the precision, recall, specificity, sensitivity, f1,
# and accuracy
# Returns a dict
def precision_recall_etc(classified_sentiment, actual_sentiment):
    if len(classified_sentiment) != len(actual_sentiment):  # if lengths don't match
        raise RuntimeError("Lengths of arguments to accuracy_percentage not the same")
    tp = fp = tn = fn = 0  # t=true, f=false, p=postive, n=negative
    for i in range(len(classified_sentiment)):
        if actual_sentiment[i] == 1:  # actual sentiment is positive
            if classified_sentiment[i] == actual_sentiment[i]:  # if matches
                tp += 1
            else:  # if doesn't match
                fn += 1
        else:  # actual sentiment is negative
            if classified_sentiment[i] == actual_sentiment[i]:  # if matches
                tn += 1
            else:  #if doesn't match
                fp += 1

    # calculate the various performance metrics
    precision = float(tp)/float(tp + fp)
    recall = float(tp)/float(tp + fn)
    specificity = float(tn)/float(fp + tn)
    NPV = float(tn)/float(tn + fn)
    f1 = 2.*float(precision*recall)/float(precision + recall)

    return {'precision': precision, 'recall': recall,
            'specificity': specificity, 'NPV': NPV,
            'f1': f1, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': float(tp + tn)/float(tp + fp + tn + fn)}
