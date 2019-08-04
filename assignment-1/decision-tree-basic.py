import pickle
import sys
from utils import *

class LeafNode:
	def __init__(self, guess):
		self.guess = guess

	def classify(self , sample_features):
		return test(self , sample_features)

class InternalNode:
	def __init__(self, f , left , right):
		self.f = f
		self.left = left
		self.right = right
		
	def classify(self , sample_features):
		return test(self , sample_features)


def test(tree , sample_features):
	if isinstance(tree, LeafNode):
		return tree.guess
		
	elif isinstance(tree, InternalNode):
		f = tree.f
		
		feature_name = f[0]
		feature_val  = f[1]
		if feature_name in sample_features:
			if sample_features[feature_name] == feature_val:
				# f=yes in test point
				return tree.right.classify(sample_features)
		
		# f=no in test point
		return tree.left.classify(sample_features)

##############################################################################

def train(labeled_samples, remaining_feature_value_pairs, remaining_depth):

	global most_freq_label_in_trainset
	if len(labeled_samples) == 0:
		return LeafNode(most_freq_label_in_trainset)

	remaining_features = remaining_feature_value_pairs
	
	label_lst = get_labels_list(labeled_samples)
	
	guess = get_most_freq_label(label_lst)

	if len(set(label_lst)) == 1:
		return LeafNode(guess)
	
	elif len(remaining_features) == 0:
		return LeafNode(guess)
		
	elif remaining_depth == 0:
		return LeafNode(guess)
		
	else:
		scores = {}
		for pair in remaining_features:
			YES , NO = get_YesNo_subset ( labeled_samples , pair)
			scores[pair] = majority_vote_count(NO) + majority_vote_count(YES)
		
		feature_with_max_score = get_feature_with_max_score(scores)
		YES , NO = get_YesNo_subset ( labeled_samples , feature_with_max_score)
		
		remaining_features.remove(feature_with_max_score)
		left  = train(NO , remaining_features , remaining_depth - 1)
		right = train(YES, remaining_features , remaining_depth - 1)
		return InternalNode(feature_with_max_score, left, right)


if __name__ == '__main__':
	# read command-line parameters
	dataset = pickle.load(open(sys.argv[1], "rb"))
	max_depth = int(sys.argv[2])
	
	label_lst = get_labels_list(dataset.training_set)
	most_freq_label_in_trainset = get_most_freq_label(label_lst)

	print("Training...")
	model = train(dataset.training_set, dataset.feature_value_pairs(), max_depth)
	print("Training complete.")

	tr_n = len(dataset.training_set)
	va_n = len(dataset.validation_set)
	te_n = len(dataset.testing_set)

	print("\nEvaluating...")
	tr_acc = accuracy(model, dataset.training_set)
	va_acc = accuracy(model, dataset.validation_set)
	te_acc = accuracy(model, dataset.testing_set)
	print("Evaluation complete:")
	print("  Training:    %s/%s: %.2f%%" % (tr_acc, tr_n, 100 * tr_acc / tr_n))
	print("  Validation:  %s/%s: %.2f%%" % (va_acc, va_n, 100 * va_acc / va_n))
	print("  Testing:     %s/%s: %.2f%%" % (te_acc, te_n, 100 * te_acc / te_n))

    
    
