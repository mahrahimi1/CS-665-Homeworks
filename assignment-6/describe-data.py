import pickle
import sys
from utils import *

################################################################################

def print_label_distribution(samples):
    h = histogram(list(sample.label for sample in samples))
    def get_key(kv):
        try:
            return int(kv[0])
        except ValueError:
            return kv[0]
    for (k, v) in sorted(h.items(), key=lambda kv: get_key(kv)):
        print("      Label %s: %s" % (k, v))

if __name__ == '__main__':
    dataset = pickle.load(open(sys.argv[1], "rb"))

    print("Dataset description:")
    print("  Training set:   %s observations" % len(dataset.training_set))
    print("    Label distribution:")
    print_label_distribution(dataset.training_set)
    print("  Validation set: %s observations" % len(dataset.validation_set))
    print("    Label distribution:")
    print_label_distribution(dataset.validation_set)
    print("  Testing set:    %s observations" % len(dataset.testing_set))
    print("    Label distribution:")
    print_label_distribution(dataset.testing_set)
          
