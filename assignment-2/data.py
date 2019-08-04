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

class DatasetNumericalMetadata(object):
    def __init__(self, dims):
        self.dims = dims
   
class Dataset(object):
    def __init__(self, training, validation, testing,
                 metadata):
        self.training_set = training
        self.validation_set = validation
        self.testing_set = testing
        self.metadata = metadata
        self.is_numerical = False
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
    def convert_to_numerical(self):
        if self.is_numerical:
            return self
        features = list(self.metadata.feature_values.keys())
        feature_values = dict((k, sorted(list(f)))
                              for (k, f) in self.metadata.feature_values.items())
        def convert_row(row):
            result = []
            for f in features:
                v = row.features[f]
                values = feature_values[f]
                one_hot = [0] * len(values)
                one_hot[values.index(v)] = 1
                result.extend(one_hot)
            return LabeledSample(row.label, result)
        new_tr = list(convert_row(v) for v in self.training_set)
        new_v =  list(convert_row(v) for v in self.validation_set)
        new_te = list(convert_row(v) for v in self.testing_set)
        new_md = DatasetNumericalMetadata(len(new_tr[0].features))
        result = NumericalDataset(new_tr, new_v, new_te, new_md)
        return result

class NumericalDataset(Dataset):
    def __init__(self, *args):
        Dataset.__init__(self, *args)
        self.is_numerical = True

def parse_line(line, feature_names):
    entries = line.decode("ascii").strip().split(",")
    label = entries[0]
    features = dict((k, v) for (k, v)
                    in zip(feature_names, entries[1:]))
    return LabeledSample(label, features)

def parse_numerical_line(line):
    entries = line.decode("ascii").strip().split(",")
    label = entries[0]
    features = list(float(i) for i in entries[1:])
    return LabeledSample(label, features)

def read_dataset_as_numeric(f):
    all_input_samples = list(parse_numerical_line(l) for l in f)
    n = len(all_input_samples)
    d = len(all_input_samples[0].features)

    random.shuffle(all_input_samples)

    training   = all_input_samples[:n//2]
    validation = all_input_samples[n//2:3*n//4]
    test       = all_input_samples[3*n//4:]

    return NumericalDataset(training, validation, test,
                            DatasetNumericalMetadata(d))

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
