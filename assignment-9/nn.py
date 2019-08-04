#!/usr/bin/env python3

import math
import random
import pickle
import sys

from autodiff import *
from data import *
from utils import *

##############################################################################

def argmax(lst):
    max_ix = 0
    max_value = lst[0]
    for i, v in enumerate(lst):
        if v > max_value:
            max_ix = i
            max_value = v
    return max_ix

class NN(object):

    def __init__(self):
        
        self.ll1 = LinearLayer(28*28 , 20)
        self.ll2 = LinearLayer(20 , 20)
        self.ll3 = LinearLayer(20 , 10)

        self.input_minibatch = zero_vector(28 * 28)
        self.onehot_label = zero_vector(10)
        self.network_prediction = self.apply_nn(self.input_minibatch)
        self.loss = cross_entropy_loss(self.network_prediction, self.onehot_label)
        
    def apply_nn(self, in_vec):
        # To apply a layer to an input vector, use the '*' operator:
        # output = layer * input
        output_ll1 = ReLU(self.ll1 * in_vec)
        output_ll2 = ReLU(self.ll2 * output_ll1)
        output_ll3 = ReLU(self.ll3 * output_ll2)
        output_ll3 = softmax(output_ll3)
        return output_ll3

    def gradient_descent_step(self, labeled_sample):
        # 1) set the values of the input minibatch to the features in labeled_sample
        # 2) set the one-hot version of the label to the correct value
        # 3) evaluate the loss function and gradient with respect to loss
        # 4) for each layer in your network, take a step in direction of the
        #    negative gradient wrt the loss

        # Note that the LinearLayer objects are defined as a list of
        # rows (layer.rows[0], layer.rows[1], etc), and each row is a list of vals:
        # (layer.rows[0].vals[0], layer.rows[0].vals[1], etc)

        # remember that in reverse mode automatic differentiation, the gradient of the
        # evaluated expression wrt the variable is stored together with the value of
        # the variable.

        # in other words,
        # var.value gives the value, var.gradient_value gives the gradient

        # Note, finally, that you won't have to call apply_nn()
        # here. That computation graph has already been defined on
        # __init__(). All you have to do is set the variable values
        # and call clear_eval() followed by evaluate() and backward()
        
        for i in range(len(self.input_minibatch.vals)):
            self.input_minibatch[i].value = labeled_sample.features[i]
        
        for i in range(len(self.onehot_label.vals)):
            if (i == labeled_sample.label):
                self.onehot_label[i].value = 1
            else:
                self.onehot_label[i].value = 0
        
        self.loss.clear_eval()
        self.loss.evaluate()
        self.loss.backward()
        
        alpha = 0.01
        for neuron in self.ll3.rows:
            for weight in neuron.vals:
                weight.value -= weight.gradient_value * alpha

        for neuron in self.ll2.rows:
            for weight in neuron.vals:
                weight.value -= weight.gradient_value * alpha

        for neuron in self.ll1.rows:
            for weight in neuron.vals:
                weight.value -= weight.gradient_value * alpha
        

    def evaluate_sample(self, features):
        for var, feature_value in zip(self.input_minibatch, features):
            var.value = feature_value
        self.network_prediction.clear_eval()
        return self.network_prediction.evaluate()

    def predict(self, features):
        return self.evaluate_sample(features)

    def classify(self, features):
        return argmax(self.predict(features))

if __name__ == '__main__':
    dataset = pickle.load(open("mnist-digits.pickle", "rb"))
    nn = NN()
    # epochs
    for i in range(100):
        # shuffle the training set between epochs
        random.shuffle(dataset.training_set)
        for sample in dataset.training_set:
            nn.gradient_descent_step(sample)
            print(".", end='')
            sys.stdout.flush()
        print("Epoch", (i+1), "done")
        evaluate_model(nn, dataset)
