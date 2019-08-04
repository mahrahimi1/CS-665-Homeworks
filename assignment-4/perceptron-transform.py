from utils import *
from data import *
import pickle
import sys
import random
import pylab
from transform import FeatureTransform
from perceptron import train

##############################################################################

class MysteryTransform(FeatureTransform):

    def transform_features(self, features):
        return self.transform_features_1(features)
        #return self.transform_features_2(features)

    
    def transform_features_1(self, features):
        new_features = []
        for f in features:
            new_features.append(f)

        for i in range(len(features)):
            for j in range(len(features)):
                new_features.append(features[i] * features[j])

        return new_features


    def transform_features_2(self, features):
        new_features = []
        for f in features:
            new_features.append(f)

        for i in range(len(features)):
            for j in range(len(features)):
                new_features.append(features[i] * features[j])

        for i in range(len(features)):
            for j in range(len(features)):
                for k in range(len(features)):
                    new_features.append(features[i] * features[j] * features[k])

        return new_features

        
def main():
    # notice that, for your convenience, this script hard-codes
    # the loading of "mystery-dataset.pickle", since you'll need
    # to hardcode the transformation in MysteryTransform above
    # anyway.
    if len(sys.argv) < 2:
        print("Usage: %s number_of_passes" %
              sys.argv[0])
        sys.exit(1)
    dataset = (pickle.load(open("mystery-dataset.pickle", "rb")).
               convert_labels_to_numerical("y").
               convert_features_to_numerical())
    # this is implemented in `transform.py`
    dataset = MysteryTransform().transform_dataset(dataset)
    k = int(sys.argv[1])
    model = train(dataset, k)
    evaluate_model(model, dataset)
    
if __name__ == '__main__':
    main()
