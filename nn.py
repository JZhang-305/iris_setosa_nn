import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import random
from random import randint
from neuralnetworkfunction import config_finder

# Put in dimensions [sepal-length, sepal-width, petal-length, petal-width]
dimensions = [_, _, _, _]

# Downloading and importing dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# Columns needed for testing and training data
X = dataset[['sepal-length', 'sepal-width', 'petal-length', 'petal-width']]
y = dataset['class']

# Splitting up data into testing and training data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)

# List of number of neurons and hidden layers to create classifier
configuration = config_finder(X_train, X_test, y_train, y_test, max_score=100)

# Creating the classifier
model = MLPClassifier(hidden_layer_sizes=configuration, max_iter=5000)

# Fitting the model to the data
model.fit(X_train, y_train)

# Converting dimensions from list to numpy array so that it can be
# reshaped for prediction
prediction = np.array(dimensions).reshape(1, -1)

# Predicting result of given dimensions
pred_result = model.predict(prediction)

# Just to make output cleaner
for i in pred_result:
    print(i)
