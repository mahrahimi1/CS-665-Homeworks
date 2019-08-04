from utils import *
from data import *

import pickle
import sys
import random
import math

##############################################################################

class NaiveBayes:

    # self.counts_per_label is a dictionary of dictionaries. Use
    # it to collect the necessary statistics per label + word. For example,
    # self.counts_per_label['ham']['friend'] will store the number of times
    # you've seen the word "friend" in non-spam sms messages.
    #
    # self.total_counts stores the label statistics

    def __init__(self, dataset, smoothing):
        self.words = dataset.metadata.feature_values
        self.counts_per_label = {}
        self.total_counts = {}
        self.smoothing = smoothing
        for label in dataset.metadata.label_values:
            self.counts_per_label[label] = {}
        
        self.w = []
        self.b = 0
    
    
    def collect_data_for_sample(self, sample):        
        label = sample.label
        
        for word in sample.features:
            if word in self.counts_per_label[label]:
                self.counts_per_label[label][word] += 1
            else:
                self.counts_per_label[label][word] = 1
        
        if label in self.total_counts:
            self.total_counts[label] += 1
        else:
            self.total_counts[label] = 1


    def fit(self):
        labels = list(self.total_counts)
        plus_one_label = labels[0]
        minus_one_label = labels[1]
        
        alpha = self.smoothing        
        theta_zero = (self.total_counts[plus_one_label] + alpha) / (self.total_counts[plus_one_label] + self.total_counts[minus_one_label] + (alpha*2))

        self.w = []
        self.b = math.log((theta_zero)/(1-theta_zero))
        
        for word in self.words:            
            if word in self.counts_per_label[plus_one_label]:
                c = self.counts_per_label[plus_one_label][word]
            else:
                c = 0
            theta_plus_one_d = (c + alpha ) / (self.total_counts[plus_one_label] + (alpha*2))
            
            if word in self.counts_per_label[minus_one_label]:
                c = self.counts_per_label[minus_one_label][word]
            else:
                c = 0            
            theta_minus_one_d = (c + alpha) / (self.total_counts[minus_one_label] + (alpha*2))
            
            w_d = math.log((theta_plus_one_d * (1-theta_minus_one_d)) / (theta_minus_one_d * (1-theta_plus_one_d)))
            self.w.append(w_d)
            
            self.b += math.log((1-theta_plus_one_d)/(1-theta_minus_one_d))
    
    
    def classify(self, features):
        labels = list(self.total_counts)
        plus_one_label = labels[0]
        minus_one_label = labels[1]
    
        x = []
        for word in self.words:
            if word in features:
                x.append(1)
            else:
                x.append(0)
        
        llr = dot(x , self.w) + self.b
    
        if llr>0:
            return plus_one_label
        else:
            return minus_one_label
        
        
def main():
    data = pickle.load(open("sms-spam-collection.pickle", "rb"))
    
    smoothing = 10
    
    nb = NaiveBayes(data, smoothing)
    for sample in data.training_set:
        nb.collect_data_for_sample(sample)
    nb.fit()
    evaluate_model(nb, data)

if __name__ == '__main__':
    main()
