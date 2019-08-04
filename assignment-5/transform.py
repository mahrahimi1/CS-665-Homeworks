from data import *

class FeatureTransform(object):

    def transform_dataset(self, dataset):
        if not isinstance(dataset.metadata, DatasetNumericalMetadata):
            raise Exception("only numerical datasets supported")
        new_training = list(LabeledSample(s.label, self.transform_features(s.features))
                            for s in dataset.training_set)
        new_validation = list(LabeledSample(s.label, self.transform_features(s.features))
                              for s in dataset.validation_set)
        new_testing = list(LabeledSample(s.label, self.transform_features(s.features))
                           for s in dataset.testing_set)
        result = Dataset(new_training,
                         new_validation,
                         new_testing,
                         DatasetNumericalMetadata(len(new_training[0].features)))
        result.numerical = True
        return result

    
