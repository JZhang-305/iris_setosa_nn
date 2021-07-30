import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import random
from random import randint
from neuralnetworkfunction import config_finder

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

X = dataset[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
y = dataset['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = 42)

#configuration = config_finder(X_train, X_test, y_train, y_test, max_score=100)

model = MLPClassifier(hidden_layer_sizes=[11], max_iter=5000)

model.fit(X_train, y_train)

prediction = np.array([3,2,4,1]).reshape(1, -1)

print(model.predict(prediction))

'''
y_pred = model.predict(X)
print(accuracy_score(y_pred, y))'''

