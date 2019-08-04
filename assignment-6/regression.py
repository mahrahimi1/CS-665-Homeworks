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

class RegressionPredictor(LinearModel):

    def __init__(self, dim):
        LinearModel.__init__(self, dim)
        self.loss_f = L2RegressionLoss()
        self.regularization = L2Regularization(0)
		
    #def predict(self, features):
    #    return dot(features, self.weights)
        
##############################################################################

dim = 30
def fact(x):
    r = 1
    for i in range(1, x+1):
        r *= i
    return r

def transform_sample(x):
    return [x ** i / fact(i) for i in range(dim)]
    
def make_sample():
    def mystery_function_of_x(x):
        raise Exception("not shown in the assignment code")
    x = random.random() * 10 - 5
    fx = mystery_function_of_x(x) 
    n = random.random()
    return LabeledSample(fx + n, transform_sample(x))

def main():
    # FYI, this is (partially) how regression.pickle was generated.
    # This is a dataset of 40 points, whose features are (scaled) monomials of x
    #
    # That scaling is there so gradient descent converges more easily
    #
    # total = 40
    # samples = list(make_sample() for _ in range(total))
    # data = Dataset(samples[:total//2],
    #                samples[total//2:3 * total // 4],
    #                samples[3 * total // 4:], DatasetNumericalMetadata(dim))
    # data.is_numerical = True
    # with open("regression.pickle", "wb") as f:
    #     pickle.dump(data, f)

    data = pickle.load(open("regression.pickle", "rb"))
    dim = len(data.training_set[0].features)

    # You fill this out with appropriate regularization parameters, learning rates, etc
    linear_regression = RegressionPredictor(dim)

    l = 0.1
    rate = 0.001
    linear_regression.train(data, l, rate)
	
    evaluate_model_regression(linear_regression , data)
    
if __name__ == '__main__':
    main()
    
    
