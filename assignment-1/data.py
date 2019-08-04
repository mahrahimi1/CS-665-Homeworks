import random

##############################################################################

class LabeledSample(object):
    def __init__(self, label, features):
        self.label = label
        self.features = features

class DatasetMetadata(object):
    def __init__(self, feature_values, label_values):
        self.feature_values = feature_values
        self.label_values = label_values

class Dataset(object):
    def __init__(self, training, validation, testing,
                 metadata):
        self.training_set = training
        self.validation_set = validation
        self.testing_set = testing
        self.metadata = metadata
    def feature_value_pairs(self, features_to_use=None):
        if features_to_use is None:
            features_to_use = set(f for f in self.metadata.feature_values)
        result = []
        for (f_name, f_values) in self.metadata.feature_values.items():
            if f_name not in features_to_use:
                continue
            for value in f_values:
                result.append((f_name, value))
        return result

def parse_line(line, feature_names):
    entries = line.decode("ascii").strip().split(",")
    label = entries[0]
    features = dict((k, v) for (k, v)
                    in zip(feature_names, entries[1:]))
    return LabeledSample(label, features)

def read_dataset(f, feature_names):
    all_input_samples = list(parse_line(l, feature_names) for l in f)
    n = len(all_input_samples)

    feature_values = {}
    label_values = set()

    for labeled_sample in all_input_samples:
        label_values.update(labeled_sample.label)
        for f_name, f_value in labeled_sample.features.items():
            feature_values.setdefault(f_name, set()).update(f_value)

    random.shuffle(all_input_samples)

    training   = all_input_samples[:n//2]
    validation = all_input_samples[n//2:3*n//4]
    test       = all_input_samples[3*n//4:]

    return Dataset(training, validation, test,
                   DatasetMetadata(feature_values, label_values))
