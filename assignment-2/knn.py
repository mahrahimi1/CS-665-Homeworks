from utils import *
import pickle
import sys
import math

class kNNClassification(object):

	def __init__(self, training_set, k):
		self.training_set = training_set
		self.k = k


	def get_euclidean_distance(self , instance1 , instance2):
		sum = 0.0
		for i in range(len(instance1)):
			sum = sum + math.pow((instance1[i] - instance2[i]) , 2)
		return math.sqrt(sum)


	def classify(self, instance):
		
		S = []
		for i in range(len(self.training_set)):
			sample = self.training_set[i]
			d = self.get_euclidean_distance(instance , sample.features)
			S.append((d , i))
			
		S.sort(key=lambda tup: tup[0])
		
		KNN_labels = []
		for i in range(self.k):
			idx = S[i][1]
			sample = self.training_set[idx]
			KNN_labels.append(sample.label)
		mode = max(set(KNN_labels), key=KNN_labels.count)

		return mode

##############################################################################

def train(labeled_samples, k):
	return kNNClassification(labeled_samples, k)

if __name__ == '__main__':
	dataset = pickle.load(open(sys.argv[1], "rb")).convert_to_numerical()
	k = int(sys.argv[2])

	model = train(dataset.training_set, k)
	evaluate_model(model, dataset)
