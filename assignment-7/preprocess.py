#!/usr/bin/env python

from data import *
import random
import pickle

punctuation = str.maketrans(",.:;!?()[]{}", "            ")
def process_line(l):
    """processes each SMS message into features that we can use to predict
    spam. 

    This is, generally speaking, *NOT* how you do NLP
    pre-processing, but is the simplest way for us to get from zero to
    naive Bayes.

    This procedure throws away a lot of important features such as
    capitalization, and does not do a number of important word
    transformations such as removal of stop words, lemmatization, and
    stemming.

    If you're interested in doing NLP correctly, take Dr. Surdeanu's
    course!
    """
    
    # minimally drop punctuation and standardize case
    l = l.translate(punctuation).lower().strip().split()

    label = l[0]
    
    words = l[1:] # list(w.replace(",.;!?()[]{}", "") for w in l[1:])
    features = dict((w, True) for w in words if len(w) > 0)
    print(label, words)
    return LabeledSample(label, features)

if __name__ == '__main__':
    with open("SMSSpamCollection", "r") as f:
        samples = list(process_line(l) for l in f)
        features = set()
        for sample in samples:
            for k in sample.features:
                features.add(k)
        random.shuffle(samples)
        total = len(samples)
        data = Dataset(samples[:total//2],
                       samples[total//2:3*total//4],
                       samples[3*total//4:],
                       DatasetMetadata(features, set(["ham", "spam"])))
        with open("sms-spam-collection.pickle", "wb") as fout:
            pickle.dump(data, fout)

                       
        
