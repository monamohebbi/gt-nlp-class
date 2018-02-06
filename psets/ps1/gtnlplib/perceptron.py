from collections import defaultdict
from gtnlplib.clf_base import predict,make_feature_vector

# deliverable 4.1
def perceptron_update(x,y,weights,labels):
    '''
    compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    '''

    yhat, scores = predict(x, weights, labels)
    actual = make_feature_vector(x, y)
    predicted = make_feature_vector(x, yhat)
    if len(predicted) != len(set(predicted.items() | actual.items())):
        for pair, count in predicted.items():
            actual[pair] = float(actual.get(pair, 0)) - float(count)
        for pair, count in actual.items():
            weights[pair] = float(weights.get(pair, 0)) + float(count)
    return weights

# deliverable 4.2
def estimate_perceptron(x,y,N_its):
    '''
    estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    '''

    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in range(N_its):
        print("iteration " + str(it))
        temp = 0
        for x_i,y_i in zip(x,y):
            weights = perceptron_update(x_i, y_i, weights, labels)
            temp += 1
            if temp % 400 == 0:
                print(temp)
        weight_history.append(weights.copy())
    return weights, weight_history

