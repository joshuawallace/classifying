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
from sklearn.ensemble import RandomForestClassifier as RFC
import numpy as np
import general_functions as general_f
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


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
# classifier = BernoulliNB()
classifier = RFC(n_estimators=150, criterion='entropy', n_jobs=3, verbose=0)

# various quantities to keep track of over the different k values
precision    = []
recall       = []
specificity  = []
NPV          = []
f1           = []
num_features = []

thresholds_to_try = np.linspace(.001, 0.004, 250)
#thresholds_to_try = [thresholds_to_try[2] ]
thresholds_to_try = [thresholds_to_try[0] ]
print "Value to compare against: " + str(len(bagofwords_training[0]))
for val in thresholds_to_try:
    feature_sel = VarianceThreshold(threshold=val)
    pipeline = Pipeline([('var_threshold', feature_sel),
                        ('berno', classifier)])
    pipeline.fit(bagofwords_training, sentiment_training)

    predict_test     = pipeline.predict(bagofwords_test)

    variances = pipeline.named_steps['var_threshold'].variances_
    num_features.append(len([value for value in variances if value > val]))

    output = general_f.precision_recall_etc(predict_test, sentiment_test)
    print output
    print num_features[-1]
    precision.append(output['precision'])
    recall.append(output['recall'])
    specificity.append(output['specificity'])
    NPV.append(output['NPV'])
    f1.append(output['f1'])

print "precision max: " + str(np.argmax(precision))
plt.plot(num_features, precision, label='precision')
print "recall max: " + str(np.argmax(recall))
plt.plot(num_features, recall, label='recall')
print "spec. max: " + str(np.argmax(specificity))
plt.plot(num_features, specificity, label='spec.')
print "NPV max: " + str(np.argmax(NPV))
plt.plot(num_features, NPV, label='NPV')
print "f1 max: " + str(np.argmax(f1))
plt.plot(num_features, f1, label='f1')

plt.legend(loc='best')
plt.xlabel("k-value")
plt.ylabel("fraction")
plt.savefig('pdf/rf_3_varthresh.pdf')


