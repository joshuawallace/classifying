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
from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2 # f_classif, mutual_info_classif
from sklearn.feature_selection import f_classif
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
classifier = BernoulliNB()
#k_to_try = np.arange(549, 515, -1)
#k_to_try = np.arange(315,295,-1) #These three are for chi2 and f_classif
#k_to_try = np.arange(200,20,-5)
#
#
k_to_try = np.arange(800,40,-20)
np.append(k_to_try, 812)
print k_to_try
print "Value to compare against: " + str(len(bagofwords_training[0]))
for val in k_to_try:
    feature_sel = SelectKBest(score_func=f_classif, k=val)
    pipeline = Pipeline([('select', feature_sel),
                        ('berno', classifier)])
    pipeline.fit(bagofwords_training, sentiment_training)

    predict_training = pipeline.predict(bagofwords_training)
    predict_test     = pipeline.predict(bagofwords_test)

    print ""
    # variances = pipeline.named_steps['var_threshold'].variances_
    # print "Number of features: " + str(len([value for value in variances if value > val]))
    print val
    training_percentage = general_f.accuracy_percentage(predict_training, sentiment_training)
    print "Training percentage for " + str(val) + " : " + str(training_percentage)
    test_percentage     = general_f.accuracy_percentage(predict_test, sentiment_test)
    print "Test percentage for " + str(val) + " :     " + str(test_percentage)



