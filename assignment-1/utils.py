from collections import defaultdict

def accuracy(model, sample_set):
    accuracy = 0
    for sample in sample_set:
        if model.classify(sample.features) == sample.label:
            accuracy += 1
    return accuracy

def histogram(values):
    result = defaultdict(int)
    for v in values:
        result[v] = result[v] + 1
    return result

def argmax(hist):
    max_value = 0
    max_key = None
    for (k, v) in hist.items():
        if v > max_value:
            max_key = k
            max_value = v
    return max_key

def majority_vote_count(labeled_samples):
    hist = histogram(sample.label for sample in labeled_samples)
    return max((v for v in hist.values()), default=0)

def majority_vote(labeled_samples):
    hist = histogram(sample.label for sample in labeled_samples)
    return argmax(hist)


def get_labels_list(labeled_samples):
	label_lst = []
	for sample in labeled_samples:
		label_lst.append(sample.label)
	return label_lst


def get_most_freq_label(label_lst):
	most_freq_label = max(set(label_lst), key=label_lst.count)
	return most_freq_label
	
	
def get_YesNo_subset ( labeled_samples , pair):
	YES = []
	NO = []
	feature_name = pair[0]
	feature_val  = pair[1]
	
	for sample in labeled_samples:
		if feature_name in sample.features:
			if sample.features[feature_name] == feature_val:
				YES.append(sample)
			else:
				NO.append(sample)
		else:
			NO.append(sample)
	
	return YES , NO


def get_feature_with_max_score(scores):
	feature_with_max_score =  max(scores, key=scores.get)
	return feature_with_max_score
	
