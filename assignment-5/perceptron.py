from utils import *
import pickle
import sys
import random
import pylab

##############################################################################

class Perceptron(object):

    def __init__(self, d):
        self.weights = [0] * d
        self.bias = 0
        self.dim = d

    def predict(self, features):
        a = 0
        for i in range(self.dim):
            a = a + (self.weights[i] * features[i])
        a = a + self.bias
        return a

        
    def classify(self, features):
        a = self.predict(features)
        if a >= 0:
            return +1
        else:
            return -1


    # NB: this assumes that the label for labeled_sample is -1 or +1!
    def update(self, labeled_sample):
        a = 0
        for i in range(self.dim):
            a = a + (self.weights[i] * labeled_sample.features[i])
        a = a + self.bias
        
        y = labeled_sample.label
        if (y * a) <= 0:
            for i in range(self.dim):
                self.weights[i] = self.weights[i] + (y * labeled_sample.features[i])
            self.bias = self.bias + y

##############################################################################

def train(dataset, k):
    labeled_samples = dataset.training_set.copy()
    model = Perceptron(len(labeled_samples[0].features))
    for i in range(k):
        random.shuffle(labeled_samples)
        for sample in labeled_samples:
            model.update(sample)
    return model
