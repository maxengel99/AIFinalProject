#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:40:27 2019

@author: alexjreed7
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


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

input_array = []
counter = 0
for x in f_feats_wout_missing_data['DEPARTURE_DELAY']:
    if x == 1:
        counter = counter + 1
        input_array.append([1]) #to categorical?
    else:
        counter = counter + 1
        input_array.append([0])
        
        
#NEED SPLITTING OF TEST AND TRAINING
        
        
        
        
input_array = np.array(input_array)

features = f_feats_wout_missing_data.drop(['DEPARTURE_DELAY'], 1)
labels =  pd.DataFrame(input_array)


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(features, input_array.ravel()) #need input_array as numpy array
neigh.score(features, labels)





