from utils import *
import perceptron

import pickle
import sys
import random
import pylab
import data

##############################################################################
class Binary_Tournament_TreeNode(object):
    def __init__(self, data , lbls):
        self.left = None
        self.right = None
        self.f = data
        self.labels = lbls


class Binary_Tournament(object):
    def __init__(self):
        self.tree = None
        self.labels = []


    def classify(self, features):
        return self.predict(features , self.tree)


    def predict(self, features, tree_node):

        if len(tree_node.labels) == 1:
            return tree_node.labels[0]

        binary_model = tree_node.f
        res = binary_model.classify(features)

        if res == (+1):
            return self.predict(features , tree_node.left)

        else:
            return self.predict(features , tree_node.right)


def build_bt_tree(labels , multiclass_dataset , k):
    if len(labels) == 1:
        return Binary_Tournament_TreeNode(None , labels)

    break_idx = len(labels) // 2
    l = labels[0 : break_idx]
    r = labels[break_idx : len(labels)]

    D_l = []
    D_r = []
    for labeledSample in multiclass_dataset.training_set:
        if labeledSample.label in l:
            s = data.LabeledSample(+1 , labeledSample.features)
            D_l.append(s)

        elif labeledSample.label in r:
            s = data.LabeledSample(-1 , labeledSample.features)
            D_r.append(s)

    D = D_l + D_r
    D_dataset = data.NumericalDataset( D , multiclass_dataset.validation_set  , multiclass_dataset.testing_set , multiclass_dataset.metadata)

    binary_model = perceptron.train(D_dataset, k)

    node = Binary_Tournament_TreeNode(binary_model , labels)
    node.left  = build_bt_tree(l , multiclass_dataset , k)
    node.right = build_bt_tree(r , multiclass_dataset , k)

    return node


def bt_train(multiclass_dataset , k):
    # Write this code! It should return a model with the same API as your
    # previous models, but should use models obtained from `perceptron.train`

    # your general strategy will be to create transformations that
    # will convert your multiclass dataset to a number of different
    # two-class datasets, train these, and then at test time, you
    # will need to run the binary classifiers, combine their results
    # appropriately, and produce a final preduction.

    model = Binary_Tournament()

    labels = []
    for labeledSample in multiclass_dataset.training_set:
        labels.append(labeledSample.label)
    model.labels = list(set(labels))

    model.tree = build_bt_tree(model.labels , multiclass_dataset , k)

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
    model = bt_train(dataset , k)

    # note that this only evaluates accuracy! You'll need to write your own code
    # to compute the confusion matrix and report that separately
    evaluate_model(model, dataset)

if __name__ == '__main__':
    main()
