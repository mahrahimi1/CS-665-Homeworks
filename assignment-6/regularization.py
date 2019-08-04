from utils import *
from data import *

##############################################################################

class L2Regularization(object):
    def __init__(self, l):
        self.l = l
    
    def loss(self, weight_vector):
        squared_norm = 0
        for i in range(len(weight_vector)):
            squared_norm = squared_norm + (weight_vector[i]**2)
        return (self.l * squared_norm)
        
    def gradient(self, weight_vector):
        g = [self.l * i for i in weight_vector]
        return g
