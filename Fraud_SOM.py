#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:41:13 2019

@author: sachin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib auto

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

from minisom import MiniSom
som = MiniSom(x = 10, y = 10,
              input_len = 15,
              sigma = 1.0,
              learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X,
                 num_iteration = 100)

# Visualizing the Self Organizing Maps
from pylab import bone, pcolor, plot, show, colorbar
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding Frauds.
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2,8)], # Mapping coordinates based on my 'mean of inter-neuron distance'.
                         mappings[(3,6)]), axis = 0) 
frauds = sc.inverse_transform(frauds)

# Combining Unsupervised Model and Supervised Model.

customers = dataset.iloc[:, 1:].values
is_fraud = np.zeros(len(dataset))

for i in range(len(dataset)):
    if dataset['CustomerID'][i] in frauds[:,0]:
        is_fraud[i] = 1

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 2,
                     kernel_initializer = 'glorot_normal',
                     activation = 'relu',
                     input_dim = 15))

classifier.add(Dense(units = 1,
                     kernel_initializer = 'glorot_normal',
                     activation = 'sigmoid'))

classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

classifier.fit(customers, is_fraud, batch_size = 1, epochs = 2)

# Probabilities of Frauds.
predictions = classifier.predict(customers)

predict_cust_fraud = np.concatenate((dataset.iloc[:,0:1].values, predictions), axis = 1) 

predict_cust_fraud = predict_cust_fraud[predict_cust_fraud[:, 1].argsort()]

predict_cust_fraud = pd.DataFrame(predict_cust_fraud,
                                  columns = ['customerID', 'Fraud-Probability'])


