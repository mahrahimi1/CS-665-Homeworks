from utils import *
from data import *
import math

##############################################################################

class L2RegressionLoss(object):
    def loss(self, prediction, sample):
        y = sample.label
        l = (prediction - y)**2
        return l
        
    def gradient(self, prediction, sample):
        x = sample.features
        y = sample.label
        g = [2 * (y-prediction) * (-float(i)) for i in x]
        return g

class LogisticLoss(object):
    def loss(self, prediction, sample):
        y = sample.label
        l = (1/math.log10(2)) * math.log10(1+(math.e ** ((-y)*prediction)))
        return l
    
    def gradient(self, prediction, sample):
        y = sample.label
        x = sample.features

        t = math.e ** ((-y)*prediction)
        t = (1/math.log10(2)) * (1 / (1+t)) * t
        g = [(t * (-y) * float(i)) for i in x]
        
        return g


class HingeLoss(object):
    def loss(self, prediction, sample):
        y = sample.label
        return max( 0 , 1- y*prediction)

    def gradient(self, prediction, sample):
        x = sample.features
        y = sample.label
        if y * prediction > 1:
            return [0.0] * len(x)
        else:
            return [-y * float(i) for i in x]
