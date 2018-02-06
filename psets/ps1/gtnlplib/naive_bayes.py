from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict

# deliverable 3.1
def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    corpus_count = defaultdict(lambda:0)
    for i in range(len(y)):
        if y[i] == label:
            for word, count in x[i].items():
                corpus_count[word] += count
    return corpus_count

# deliverable 3.2
def estimate_pxy(x,y,label,smoothing,vocab):
    '''
    Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    '''
    probabilities = defaultdict(float)
    count = get_corpus_counts(x, y, label)
    for v in vocab:
        probabilities[v] = estimate_helper(v, count, smoothing, vocab)
    return probabilities

def estimate_helper(x_i, count, alpha, vocab):
    num = alpha + count[x_i]
    denom = alpha * len(vocab)
    for v in vocab:
        denom += count[v]
    return np.log(num/denom)

# deliverable 3.3
def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    
    labels = set(y)
    doc_counts = defaultdict(float)
    counts = defaultdict(float)
    label_dict = {}
    vocab = set()

    for x_i in x:
        vocab = vocab.union(set(x_i))

    vocab = list(vocab)

    for y_i in y:
        label_dict[y_i] = float(label_dict.get(y_i, 0) + 1)

    for label in labels:
        counts = estimate_pxy(x, y, label, smoothing, vocab)
        # print(sum(np.exp(list(counts.values()))))
        for word, prob in counts.items():
            
            doc_counts[(label, word)] = prob

    return doc_counts

# deliverable 3.4
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    '''
    find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values
    :returns: best smoothing value
    :rtype: float

    '''

    raise NotImplementedError







