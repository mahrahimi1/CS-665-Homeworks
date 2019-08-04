from utils import *
import perceptron

import pickle
import sys
import random
import pylab

##############################################################################

class OVA(object):

    def __init__(self):
        self.binary_models = []
        self.labels = []

    def classify(self, features):

        num_of_labels = len(self.labels)
        scores = [0] * num_of_labels

        for i in range(num_of_labels):
            binary_model = self.binary_models[i]
            y = binary_model.classify(features)
            scores[i] = scores[i] + y

        idx = scores.index(max(scores))
        return self.labels[idx]


def ova_train(multiclass_dataset , k):
    # Write this code! It should return a model with the same API as your
    # previous models, but should use models obtained from `perceptron.train`

    # your general strategy will be to create transformations that
    # will convert your multiclass dataset to a number of different
    # two-class datasets, train these, and then at test time, you
    # will need to run the binary classifiers, combine their results
    # appropriately, and produce a final preduction.

    model = OVA()
	
    labels = []
    for labeledSample in multiclass_dataset.training_set:
        labels.append(labeledSample.label)
    model.labels = list(set(labels))
	
    for i in range(len(model.labels)):
        dataset = multiclass_dataset.convert_labels_to_numerical(model.labels[i])
        binary_model = perceptron.train(dataset, k)
        model.binary_models.append(binary_model)

    return model


def main():
    if len(sys.argv) < 3:
        print("Usage: %s dataset number_of_passes" %
              sys.argv[0])
        sys.exit(1)

    # notice that `dataset` will not have numerical +1/-1 labels, and
    # different reductions will require different transformations.
    dataset = (pickle.load(open(sys.argv[1], "rb")).
               convert_features_to_numerical())

    k = int(sys.argv[2])
    model = ova_train(dataset , k)

    # note that this only evaluates accuracy! You'll need to write your own code
    # to compute the confusion matrix and report that separately
    evaluate_model(model, dataset)

if __name__ == '__main__':
    main()
