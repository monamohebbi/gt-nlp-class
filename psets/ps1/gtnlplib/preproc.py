from collections import Counter

import pandas as pd
import numpy as np

# deliverable 1.1
def bag_of_words(text):
    '''
    Count the number of word occurences for each document in the corpus

    :param text: a document, as a single string
    :returns: a Counter for a single document
    :rtype: Counter
    '''
    words = text.split()
    return Counter(words)

# deliverable 1.2
def aggregate_counts(bags_of_words):
    '''
    Aggregate word counts for individual documents into a single bag of words representation

    :param bags_of_words: a list of bags of words as Counters from the bag_of_words method
    :returns: an aggregated bag of words for the whole corpus
    :rtype: Counter
    '''
    # YOUR CODE GOES HERE
    counts = Counter()

    for lyric in bags_of_words:
        counts.update(lyric)
    return counts

# deliverable 1.3
def compute_oov(bow1, bow2):
    '''
    Return a set of words that appears in bow1, but not bow2

    :param bow1: a bag of words
    :param bow2: a bag of words
    :returns: the set of words in bow1, but not in bow2
    :rtype: set
    '''
    return set(bow1) - set(bow2)

# deliverable 1.4
def prune_vocabulary(training_counts, target_data, min_counts):
    '''
    prune target_data to only words that appear at least min_counts times in training_counts

    :param training_counts: aggregated Counter for training data
    :param target_data: list of Counters containing dev bow's
    :returns: new list of Counters, with pruned vocabulary
    :returns: list of words in pruned vocabulary
    :rtype: list of Counters, set
    '''
    vocab = set()
    for word in training_counts:
        if training_counts[word] >= min_counts:
            vocab.add(word)
    i = 0
    pruned_target_data = []
    for counter in target_data:
        pruned_counter = Counter()
        for word, count in counter.items():
            if word in vocab:
                pruned_counter[word] = count
        pruned_target_data.append(pruned_counter)

    vocab = list(vocab)
    return pruned_target_data, vocab

# deliverable 5.1
def make_numpy(bags_of_words, vocab):
    '''
    Convert the bags of words into a 2D numpy array

    :param bags_of_words: list of Counters
    :param vocab: pruned vocabulary
    :returns: the bags of words as a matrix
    :rtype: numpy array
    '''
    vocab = sorted(vocab)
    height = len(bags_of_words)
    width = len(vocab)
    ndArray = np.zeros((height, width))
    for i in range(len(bags_of_words)):
        for word,count in bags_of_words[i].items():
            try:
                index = vocab.index(word)
            except ValueError:
                index = -1
            if index >= 0:
                ndArray[i][index] = count
    return ndArray
### helper code

def read_data(filename,label='Era',preprocessor=bag_of_words):
    df = pd.read_csv(filename)
    return df[label].values,[preprocessor(string) for string in df['Lyrics'].values]

def oov_rate(bow1,bow2):
    return len(compute_oov(bow1,bow2)) / len(bow1.keys())

