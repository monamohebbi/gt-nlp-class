from gtnlplib.constants import OFFSET
import numpy as np
import torch
import operator
# deliverable 6.1
def get_top_features_for_label_numpy(weights,label,k=5):
    '''
    Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in
    :returns: list of tuples of features and weights
    :rtype: list
    '''

    top_features = []

    filtered_weights = {pair : weight for pair, weight in weights.items() if pair[0] == label}

    for pair, weight in weights.items():
        if pair[0] == label:
            if len(top_features) < k:
                top_features.append((pair, weight))
            else:
                min_feature = min(top_features, key=lambda x:x[1])
                if weight > min_feature[1]:
                    top_features = [(pair, weight) if x == min_feature else x for x in top_features]

    top_features.sort(key=operator.itemgetter(1), reverse=True)
    return top_features


# deliverable 6.2
def get_top_features_for_label_torch(model,vocab,label_set,label,k=5):
    '''
    Return the five words with the highest weight for a given label.

    :param model: PyTorch model
    :param vocab: vocabulary used when features were converted
    :param label_set: set of ordered labels
    :param label: the label you are interested in 
    :returns: list of words
    :rtype: list
    '''

    vocab = sorted(vocab)
    features = list(model.parameters())[0]

    label_dict = {}
    i = 0
    for l in label_set:
        label_dict[l] = i
        i = i + 1

    features = (list(model.parameters())[0][label_dict[label]].data.numpy())
    features = features.argsort()[-k:][::-1]
    ret = []
    for feat_i in features:
        ret.append(vocab[feat_i])

    return ret

# deliverable 7.1
def get_token_type_ratio(counts):
    '''
    compute the ratio of tokens to types

    :param counts: bag of words feature for a song, as a numpy array
    :returns: ratio of tokens to types
    :rtype: float

    '''
    # print(counts)
    distinct = 0
    sum = 0
    for count in counts:
        if count != 0:
            distinct += 1
            sum += count
    if distinct != 0:
        return sum / distinct
    return 0

# deliverable 7.2
def concat_ttr_binned_features(data):
    '''
    Discretize your token-type ratio feature into bins.
    Then concatenate your result to the variable data

    :param data: Bag of words features (e.g. X_tr)
    :returns: Concatenated feature array [Nx(V+7)]
    :rtype: numpy array

    '''
    num_bins = 7
    include_bins = []
    for row in data:
        # print(str(row.shape) + str(r))
        ratio = int(get_token_type_ratio(row))
        bins = np.asarray([1 if i == ratio else 0 for i in range(num_bins)])
        np.concatenate((row, bins), axis=0)
        new_row = np.append(row, bins)
        include_bins.append(new_row)
    return np.asarray(include_bins)


