import sys
import collections
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
import nltk
import random
random.seed(0)

from collections import Counter

from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")		  # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
	print "python sentiment.py <path_to_data> <0|1>"
	print "0 = NLP, 1 = Doc2Vec"
	exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
	train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
	
	if method == 0:
		train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
		nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
	if method == 1:
		train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
		nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
	print "Naive Bayes"
	print "-----------"
	evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
	print ""
	print "Logistic Regression"
	print "-------------------"
	evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
	"""
	Loads the train and test set into four different lists.
	"""
	train_pos = []
	train_neg = []
	test_pos = []
	test_neg = []
	with open(path_to_dir+"train-pos.txt", "r") as f:
		for i,line in enumerate(f):
			words = [w.lower() for w in line.strip().split() if len(w)>=3]
			train_pos.append(words)
	with open(path_to_dir+"train-neg.txt", "r") as f:
		for line in f:
			words = [w.lower() for w in line.strip().split() if len(w)>=3]
			train_neg.append(words)
	with open(path_to_dir+"test-pos.txt", "r") as f:
		for line in f:
			words = [w.lower() for w in line.strip().split() if len(w)>=3]
			test_pos.append(words)
	with open(path_to_dir+"test-neg.txt", "r") as f:
		for line in f:
			words = [w.lower() for w in line.strip().split() if len(w)>=3]
			test_neg.append(words)

	return train_pos, train_neg, test_pos, test_neg

def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
	"""
	Returns the feature vectors for all text in the train and test datasets.
	"""
	# English stopwords from nltk
	stopwords = set(nltk.corpus.stopwords.words('english'))
	
	# Determine a list of words that will be used as features. 
	# This list should have the following properties:
	#   (1) Contains no stop words
	#   (2) Is in at least 1% of the positive texts or 1% of the negative texts
	#   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
	# YOUR CODE HERE
	train_pos_dict, features_pos = filter_part1_part2(train_pos, stopwords)
	train_neg_dict, features_neg = filter_part1_part2(train_neg, stopwords)

	features12 = features_pos + features_neg
	common_words = train_pos_dict.viewkeys() & train_neg_dict.viewkeys()
	'''features12 = features_pos.viewkeys() | features_neg.viewkeys()
	common_words = train_pos.viewkeys() & train_neg.viewkeys()'''		
	features3 = [word for word in common_words if train_pos_dict[word] >= 2*train_neg_dict[word] or train_neg_dict[word] >= 2*train_pos_dict[word]]
			
	features = set(features12) & set(features3)
	print "after featuring"
	train_pos_vec = features_vector_mapping(features,train_pos)
	train_neg_vec = features_vector_mapping(features,train_neg)
	test_pos_vec = features_vector_mapping(features,test_pos)
	test_neg_vec = features_vector_mapping(features,test_neg)
	print "after vecting"
	
	# Using the above words as features, construct binary vectors for each text in the training and test set.
	# These should be python lists containing 0 and 1 integers.
	# YOUR CODE HERE

	# Return the four feature vectors
	return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
	"""
	Returns the feature vectors for all text in the train and test datasets.
	"""
	# Doc2Vec requires LabeledSentence objects as input.
	# Turn the datasets from lists of words to lists of LabeledSentence objects.
	# YOUR CODE HERE

	# Initialize model
	model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
	sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
	model.build_vocab(sentences)

	# Train the model
	# This may take a bit to run 
	for i in range(5):
		print "Training iteration %d" % (i)
		random.shuffle(sentences)
		model.train(sentences)

	# Use the docvecs function to extract the feature vectors for the training and test data
	# YOUR CODE HERE
	
	# Return the four feature vectors
	return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def obj_map(words_list, label):
    labeled_list = []
    for index, text in enumerate(list):
        LS_obj =  LabeledSentence(words=text, tags = [listname + str(index)])
        labeled.append(LS_obj)
    return labeled_list


def build_models_NLP(train_pos_vec, train_neg_vec):
	"""
	Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
	"""
	Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

	# Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
	# For BernoulliNB, use alpha=1.0 and binarize=None
	# For LogisticRegression, pass no parameters
	# YOUR CODE HERE
	
	nb = BernoulliNB(alpha = 1.0, binarize = None)
	train = train_pos_vec + train_neg_vec
	nb_model = nb.fit(train, Y)
	lr = LogisticRegression()
	lr_model = lr.fit(train, Y)

	return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
	"""
	Returns a GaussianNB and LosticRegression Model that are fit to the training data.
	"""
	Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

	# Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
	# For LogisticRegression, pass no parameters
	# YOUR CODE HERE

	return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
	"""
	Prints the confusion matrix and accuracy of the model.
	"""
	# Use the predict function and calculate the true/false positives and true/false negative.
	# YOUR CODE HERE
	'''
	pos_temp = model.predict(test_pos_vec)
	neg_temp = model.predict(test_neg_vec)
   	pos_predict = Counter(pos_temp)
	neg_predict = Counter(neg_temp)
	tp = pos_predict['pos']
	tn = neg_predict['neg']
	fn = pos_predict['neg']
	fp = neg_predict['pos']
	accuracy = (tp+tn)/float(tp+tn+fn+fp)'''

	tp, fn, fp, tn = 0, 0, 0, 0
	predict_pos = model.predict(test_pos_vec)
	predict_neg = model.predict(test_neg_vec)

	for each in predict_pos:
	  if each == "pos":
		tp +=1
	  else:
		fp +=1

	for each in predict_neg:
	  if each == "neg":
		tn +=1
	  else:
		fn +=1

	accuracy = (tp+tn)/float(tp+tn+fn+fp)

	if print_confusion:
		print "predicted:\tpos\tneg"
		print "actual:"
		print "pos\t\t%d\t%d" % (tp, fn)
		print "neg\t\t%d\t%d" % (fp, tn)
	print "accuracy: %f" % (accuracy)

def filter_part1_part2(words_list, stop_w):
	word_dict = {}
	for entry in words_list:	
		word_count = Counter(entry)
		for k,v in word_count.iteritems():
			if k not in stop_w:			
				word_dict[k] = word_dict.get(k, 0) + 1 
	#features_temp = {x:y for x,y in word_dict.iteritems() if y >= int(len(words_list)*0.01)}	
	features_temp = [word for word in word_dict.keys() if word_dict[word] >= int(len(words_list)*0.01)]
	return word_dict, features_temp

def features_vector_mapping(features, txtSet):
	f_vector = []
	for text in txtSet:
		temp = []
		for feature in features:
			temp.append(1 if feature in text else 0)
		f_vector.append(temp)
	return f_vector
 

if __name__ == "__main__":
	main()
