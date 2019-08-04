import random
from data import *
from utils import *
import pickle
import knn
import sys

def inner_product(v1, v2):
    return sum((a * b) for (a, b) in zip(v1, v2))

def normalize(vec):
    l = inner_product(vec, vec) ** 0.5
    return list((c / l) for c in vec)

def unit_normal(d):
    s = (1.0 / d) ** 0.5
    return list(random.normalvariate(0, s) for _ in range(d))

def transform_sample(old_sample, new_basis):
    new_features = list(inner_product(old_sample.features, v) for v in new_basis)
    return LabeledSample(old_sample.label, new_features)

def linear_project(dataset, d):
    n_features = len(dataset.training_set[0].features)

    new_basis = []
    for i in range(d):
        new_basis.append(unit_normal(n_features))

    new_training = list(transform_sample(s, new_basis) for s in dataset.training_set)
    new_validation = list(transform_sample(s, new_basis) for s in dataset.validation_set)
    new_testing = list(transform_sample(s, new_basis) for s in dataset.testing_set)
    return NumericalDataset(new_training, new_validation, new_testing,
                            DatasetNumericalMetadata(d))

if __name__ == '__main__':
    # here, k is the parameter in the kNN classifier,
    # and *d* is the parameter for reducing the dimensionality of the dataset.
    
    dataset = pickle.load(open(sys.argv[1], "rb")).convert_to_numerical()
    k = int(sys.argv[2])
    d = int(sys.argv[3])
    dataset = linear_project(dataset, d)
    model = knn.train(dataset.training_set, k)
    evaluate_model(model, dataset)
