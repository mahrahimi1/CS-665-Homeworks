from utils import *
from data import *
from linear_model import *
from regularization import *
from loss_functions import *

import pickle
import sys
import random
import math

##############################################################################

# Note that this class subclasses from LinearModel, which is the base
# class that implements the generic gradient descent. This class
# only constructs the correct instances of the loss function and regularizer.

class SVMClassification(LinearModel):

    def __init__(self, dim):
        LinearModel.__init__(self, dim)
        self.loss_f = HingeLoss()
        self.regularization = L2Regularization(0)

    def classify(self, features):
        return sign(dot(features, self.weights))

##############################################################################

def main():
    data = pickle.load(open("agaricus-lepiota.pickle", "rb"))
    data = data.convert_labels_to_numerical("e").convert_features_to_numerical(add_bias=True)
    dim = len(data.training_set[0].features)

    svm = SVMClassification(dim)
    # fill this out to train the model, evaluate it, find points in
    # the margin, etc.
    l = 0.01
    rate = 0.01

    svm.train(data, l, rate)
    evaluate_model(svm, data)

if __name__ == '__main__':
    main()
