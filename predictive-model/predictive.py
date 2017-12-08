#!/usr/bin/env python

# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Our classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pandas.read_csv('iris.csv', names=names)

print '# Iris Dataset #'

# Part 1. Summarize the dataset

# Look at the dimensions of the dataset
print('\n1. Shape: ' + str(data.shape))

print('\n2. Head\n')
print data.head(5)

print('\n3. Description\n')
print data.describe()

# Class Distribution
# Look at the number of rows belonging to each class ('Iris-setosa')
print('\n4. Group by class\n')
print(data.groupby('class').size())

# Part 2. Visualise the data with matplotlib boxplot
# Univariate graphs

# Boxplot
# data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# Histogram
# data.hist()
# plt.show()

# Multi-variate graph
# scatter_matrix(data)
# plt.show()

# Part 3. Create some models of the data

# Separate out a validation dataset
# Keep back data from algorithm
# Use data to get an independent idea of how accurate our model is
array = data.values

print array

x = array[:,0:4]
y = array[:,4]

# 20% of data for validating, 80% of data for training
validation_size = 0.20
seed = 7
xtr, xval, ytr, yval = model_selection.train_test_split(
                     x,
                     y,
                     test_size=validation_size,
                     random_state=seed)

# xtr and xval can be used for prepping models
# ytr and yval can be used later

# Estimate accuracy of model with 10 fold cross validation
# Split into 10
# Train on 9
# Test on 1
# Repeat for all splits

# Number of correct / total * 100
scoring = 'accuracy'

# Now, we build and evaluate each model
# Use six different algorithms
# LR, LDA (linear), KNN, CART, NB, SVM (non-linear)
#
# Logistic Regression (LR)
# Linear Discriminant Analysis (LDA)
# K-Nearest Neighbors (KNN)
# Classification and Regression Trees (CART)
# Gaussian Naive Bayes (NB)
# Support Vector Machines (SVM)

models = []
models.append(('LR', LogisticRegression()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, xtr, ytr, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Select your best model

# Make predictions
