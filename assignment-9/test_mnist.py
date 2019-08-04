#!/usr/bin/env python3

import pylab
import scipy
import pickle
import sys

if __name__ == '__main__':
    dataset = pickle.load(open("mnist-digits.pickle", "rb"))
    ix = int(sys.argv[1])
    print("Label: ", dataset.training_set[ix].label)
    pylab.matshow(scipy.array(dataset.training_set[ix].features).reshape(28, 28))
    pylab.show()

    
