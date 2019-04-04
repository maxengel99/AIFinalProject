#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:55:18 2019

@author: alexjreed7
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers
from keras.utils import to_categorical
#y_binary 

df = pd.read_csv('./Data/added_weather_fields.csv')[['DAY_OF_WEEK', 'MONTH', 'DAY', 'DISTANCE', 'ORIGIN_AVG_VISIBILITY', 'DESTINATION_AVG_VISIBILITY', 'DESTINATION_AVG_WIND', 'ORIGIN_AVG_WIND', 'DESTINATION_SNOW_CM', 'ORIGIN_SNOW_CM', 'DESTINATION_MIN_TEMPERATURE', 'ORIGIN_MIN_TEMPERATURE', 'DEPARTURE_DELAY']].copy()

f_feats_wout_missing_data = pd.DataFrame()
for j, x in df.iterrows():
    flag =  True
    for y in x:
        if not(isinstance(y, float)):
            flag = False
    if flag:
        if x['DEPARTURE_DELAY'] > 15:
            x['DEPARTURE_DELAY'] = 1
        else:
            x['DEPARTURE_DELAY'] = 0
        f_feats_wout_missing_data = f_feats_wout_missing_data.append(x)
#        print('here', x)

#f_labels = f_feats_wout_missing_data[['DEPARTURE_DELAY']].copy()

#f_labels_bin = df_labels['DEPARTURE_DELAY'].map({'on_time': 1, 'delayed': 0})

f_labels_bin = np.array([])
#
counter = 0
for x in f_feats_wout_missing_data['DEPARTURE_DELAY']:
    if x == 1:
         f_labels_bin = np.concatenate((f_labels_bin, to_categorical(1)))
    else:
         f_labels_bin = np.concatenate((f_labels_bin, to_categorical(0)))
        

f_feats_wout_missing_data = f_feats_wout_missing_data.drop(columns='DEPARTURE_DELAY')


model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))

# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu', input_shape=(12,)),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(1, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(f_feats_wout_missing_data.values, f_labels_bin, epochs=10, batch_size=32)
