# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:55:19 2023

@author: Win 10
"""
###############
#logistic regression
###############

#import library
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Reading-in the Iris data
try:
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print('From URL:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')    
except HTTPError:
    s = 'iris.data'
    print('From local Iris path:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')

###############
#Logistic regression
###############
class LogisticRegressionGD:
    """Gradient descent-based logistic regression classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after training.
    b_ : Scalar
      Bias unit after fitting.
    losses_ : list
      Mean squared error loss function values in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : Instance of LogisticRegressionGD

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            
            self.w_ += self.eta * X.T.dot(errors) / X.shape[0]   ###
    
            self.b_ += self.eta * errors.mean()
            
            loss = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) / X.shape[0]     # <<------------|
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))    # <<-----------------------------------------|

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
###########
#EXAMPLE 1: two classes, two features
###########
# Logistic regression is a model for binary classification
# work with only two classes: setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
# For simplicity of rappresentation, extract two features: sepal length and petal length
X = df.iloc[0:100, [0, 2]].values
# Plot the data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='s', label='Versicolor')

plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# Call the class
lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
# Fit the class on the (restricted) dataset
lrgd.fit(X,y);

#plot the decision boundary
plot_decision_regions(X=X, 
                      y=y,
                      classifier=lrgd)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Plot the predicted data
y_pred = lrgd.predict(X)
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='*', label='Versicolor')
dict_iris = {0: '.', 1: 'x'}
for j in np.arange(100):
    plt.scatter(X[j, 0], X[j, 1],
            color='black', marker=dict_iris[y_pred[j]])
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

###############
#EXAMPLE 3: multiclass
###############
# Using integer labels is a recommended approach to avoid technical glitches
# and improve computational performance due to a smaller memory footprint.
y_all = df.iloc[:, 4].values
y_all = np.where(y_all == 'Iris-setosa', 0, y_all)
y_all = np.where(y_all == 'Iris-versicolor', 1, y_all)
y_all = np.where(y_all == 'Iris-virginica', 2, y_all)
X_all2 = df.iloc[:, :4].values
print('Class labels:', np.unique(y_all))

probabilities = np.zeros((X_all2.shape[0], 3))
for k in range(3):
    y_k = np.where(y_all == k, 1, 0)
    lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
    lrgd.fit(X_all2, y_k)
    probabilities[:, k] = lrgd.activation(lrgd.net_input(X_all2))
    
probabilities
np.sum(probabilities, axis = 1)
np.argmax(probabilities, axis =1)
    