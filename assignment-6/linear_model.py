from utils import *
from data import *
import math

##############################################################################

class LinearModel(object):

    def __init__(self, dim):
        self.dim = dim
        self.weights = [0.0] * dim

    def predict(self, features):
        return dot(self.weights, features)

    def train(self, dataset, l, alpha=0.001):
        self.regularization.l = l

        loss_f = self.loss_f
        reg = self.regularization
        
        n_iter = 0
        
        train_size = len(dataset.training_set)
        val_size = len(dataset.validation_set)
        validation_loss = 0
        
        train_loss_list = []
        val_loss_list = []

        while True:

            g = [0.0] * self.dim
            
            for i in range(train_size):
                sample = dataset.training_set[i]
                pred = dot(sample.features, self.weights)
                g_sample = loss_f.gradient(pred , sample)
                g = [sum(x) for x in zip(g, g_sample)]
            
            g = [(i / train_size) for i in g]
            g = g + reg.gradient(self.weights)
            
            t = [alpha * i for i in g]
            self.weights = [a-b for a,b in zip(self.weights,t)]
            
            training_loss = 0
            for i in range(train_size):
                sample = dataset.training_set[i]
                pred = dot(sample.features, self.weights)
                tl_sample = loss_f.loss(pred , sample)
                training_loss = training_loss + tl_sample
            training_loss = (training_loss / train_size) + (0.5)*(reg.loss(self.weights))
            
            prev_validation_loss = validation_loss
            validation_loss = 0
            for i in range(val_size):
                sample = dataset.validation_set[i]
                pred = dot(sample.features, self.weights)
                vl_sample = loss_f.loss(pred , sample)
                validation_loss = validation_loss + vl_sample
            validation_loss = (validation_loss / val_size) + (0.5)*(reg.loss(self.weights))
            
            print("\r  Train: %.2f Validation: %.2f log10(|g|): %.2f   |g|^2: %.5f       " % (training_loss, validation_loss, math.log10(dot(g, g)) , dot(g, g)),end='')
            n_iter += 1
            
            train_loss_list.append(training_loss)
            val_loss_list.append(validation_loss)
            
            if (n_iter == 1):
                first_val_loss = validation_loss
            
            g_norm = dot(g, g)
            if (g_norm < 0.01):
                break
                
            if ((n_iter > 1) and (validation_loss<first_val_loss) and (validation_loss > prev_validation_loss)):
                break

        print()
        print("Training Loss: %.2f, Validation Loss: %.2f" %
              (training_loss, validation_loss))
        print("  n: %s" % n_iter)