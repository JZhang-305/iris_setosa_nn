# Function to find optimal configuration of neurons and hidden layers

import random
from random import randint
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def config_finder(
        X_train,
        X_test,
        y_train,
        y_test,
        max_score=90,
        max_iterations=1000):

    i = 0
    highest_score = 0
    highest_config = []
    while highest_score < max_score and i < max_iterations:
        i += 1
        randomlist = random.sample(range(1, 13), randint(1, 6))
        # How many hidden layers? How many neurons does this have?
        nnet = MLPClassifier(hidden_layer_sizes=randomlist, max_iter=5000)
        nnet.fit(X_train, y_train)

        predictions = nnet.predict(X_test)

        accuracy = accuracy_score(y_test, predictions) * 100
        if accuracy > highest_score:
            highest_score = accuracy
            highest_config = randomlist
        '''
	print(i)
        print(randomlist)
        print(accuracy)
        print('------------------')
        print(highest_score)
        print(highest_config)
        print('\n')
	'''
    return highest_config
