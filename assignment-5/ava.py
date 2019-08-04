from utils import *
import perceptron

import pickle
import sys
import random
import pylab
import data

##############################################################################
class AVA(object):

    def __init__(self):
        self.f = {}
        self.labels = []

    def classify(self, features):

        num_of_labels = len(self.labels)
        scores = [0] * num_of_labels

        for i in range(num_of_labels-1):
            for j in range(i+1 , num_of_labels):
                binary_model = self.f[(i,j)]
                y = binary_model.classify(features)
                scores[i] = scores[i] + y
                scores[j] = scores[j] - y

        idx = scores.index(max(scores))
        return self.labels[idx]


def ava_train(multiclass_dataset , k):
    # Write this code! It should return a model with the same API as your
    # previous models, but should use models obtained from `perceptron.train`

    # your general strategy will be to create transformations that
    # will convert your multiclass dataset to a number of different
    # two-class datasets, train these, and then at test time, you
    # will need to run the binary classifiers, combine their results
    # appropriately, and produce a final preduction.
	
    model = AVA()

    labels = []
    for labeledSample in multiclass_dataset.training_set:
        labels.append(labeledSample.label)
    model.labels = list(set(labels))

    for i in range(len(model.labels)-1):

        D_pos = []
        for labeledSample in multiclass_dataset.training_set:
            if labeledSample.label == model.labels[i]:
                l = data.LabeledSample(+1 , labeledSample.features)
                D_pos.append(l)

        for j in range(i+1 , len(model.labels)):
            
            D_neg = []
            for labeledSample in multiclass_dataset.training_set:
                if labeledSample.label == model.labels[j]:
                    l = data.LabeledSample(-1 , labeledSample.features)
                    D_neg.append(l)

            D_bin = D_pos + D_neg
            D_bin_dataset = data.NumericalDataset( D_bin , multiclass_dataset.validation_set  , multiclass_dataset.testing_set , multiclass_dataset.metadata)
            #D_bin_dataset.is_numerical = multiclass_dataset.is_numerical

            binary_model = perceptron.train(D_bin_dataset, k)
            model.f[(i,j)] = binary_model

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
    model = ava_train(dataset , k)

    # note that this only evaluates accuracy! You'll need to write your own code
    # to compute the confusion matrix and report that separately
    evaluate_model(model, dataset)

if __name__ == '__main__':
    main()
