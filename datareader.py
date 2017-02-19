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