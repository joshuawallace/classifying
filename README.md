
# COS 424: Sentiment Analysis Task (Spring 2017)

This was the first homework assignment for COS 424, **"Fundamentals of Machine Learning."**

---

(borrowed from the original README file provided by the instructors)

code found in github.com/joshuawallace/classifying

This directory has the following subdirectories:

- data/ where is stored the training and test data sets, stored as
   <sample num>   <sentence>  <sentiment> 
   where <sentiment> is either 0 (negative) or 1 (positive).
- output/ where is stored the processed data from the preprocessSentences.py script.

(you will also want to create a pdf directory to store plots from the feature selection scripts, or modify the save paths in those codes)

-------------------------------------

The code preprocessSentences.py is called as follows: 
	python ./preprocessSentences.py -p <data> -o <out> -v <vocab>
<data> is the path of the directory containing train.txt,
<out> is an optional argument specifying the prefix of output files, and 
<vocab> is an optional argument specifying the path to an existing vocabulary file. 

The script generates four output files in <data>: 
	- A vocabulary file 'out_vocab_*.txt’ (if one is not specified when calling the script) with tokenized words from the training data (where '*' is the word count threshold, set to default as 5 in the script), 
	- A list of training samples numbers in 'out_samples_classes_*.txt',
	- A list of sentiment labels corresponding to each training sample, 'out_classes_*.txt', and
	- A 'bag of words' featurized representation of each sentence in the training set, 'out_bag_of_words_*.csv'

----------------------------------------------

The remaining codes do not take any command line arguments and so can be called simply using `python <filename>`

The files are:

 - general_functions.py, which contains some basic functions used by all the classifier codes (performance calculation, data input)

 - nb_nltk.py, which implements the NB classifier of nltk

 - nb_sklearn.py, which implements the NB classifier of scikit-learn

 - rf_sklearn.py, which implements the random forest classifier of scikit-learn

 - featureselection_varthresh.py, which implements variance threshold feature selection with the Bernoulli NB and RF classifiers

 - featureselection_selectkbest.py, which implements select k best feature selection with the Bernoulli NB and RF classifiers

 - featureselection_recursiveelimination.py, which implements a recursive feature elimination feature selection with the Bernoulli NB and RF classifiers
