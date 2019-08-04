from collections import defaultdict

def accuracy(model, sample_set):
    accuracy = 0
    for i, sample in enumerate(sample_set):
        if i % 100 == 0:
            print("\r       \r%s" % i, end='')
        if model.classify(sample.features) == sample.label:
            accuracy += 1
    print()
    return accuracy

def histogram(values):
    result = defaultdict(int)
    for v in values:
        result[v] = result[v] + 1
    return result

def mode(hist):
    max_value = 0
    max_key = None
    for (k, v) in hist.items():
        if v > max_value:
            max_key = k
            max_value = v
    return max_key

def argmax_list(key_value_list):
    max_value = None
    max_key = None
    for (k, v) in key_value_list:
        if max_value == None or v > max_value:
            max_value = v
            max_key = k
    return max_key

def majority_vote_count(labeled_samples):
    hist = histogram(sample.label for sample in labeled_samples)
    return max((v for v in hist.values()), default=0)

def majority_vote(labeled_samples):
    hist = histogram(sample.label for sample in labeled_samples)
    return mode(hist)

def evaluate_model(model, dataset):
    print("\nEvaluating...")
    training_accuracy = accuracy(model, dataset.training_set)
    validation_accuracy = accuracy(model, dataset.validation_set)
    testing_accuracy = accuracy(model, dataset.testing_set)
    print("Evaluation complete:")
    print("  Training:    %s/%s: %.2f%%" % (training_accuracy, len(dataset.training_set), 100 * training_accuracy / len(dataset.training_set)))
    print("  Validation:  %s/%s: %.2f%%" % (validation_accuracy, len(dataset.validation_set), 100 * validation_accuracy / len(dataset.validation_set)))
    print("  Testing:     %s/%s: %.2f%%" % (testing_accuracy, len(dataset.testing_set), 100 * testing_accuracy / len(dataset.testing_set)))
    
