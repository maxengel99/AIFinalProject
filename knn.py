#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:40:27 2019

@author: alexjreed7
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Data/combined_5050.csv')[['MONTH', 'DAY_OF_WEEK', 'DISTANCE', "SCHEDULED_DEPARTURE", 'DEPARTURE_DELAY']].copy()
f_feats_wout_missing_data = pd.DataFrame()
for j, x in df.iterrows():
    flag =  True
    for y in x:
        if not(isinstance(y, float) or not(isinstance(y, str))):
            flag = False
    if flag:
        if x['DEPARTURE_DELAY'] > 15:
            x['DEPARTURE_DELAY'] = 1
        else:
            x['DEPARTURE_DELAY'] = 0
        f_feats_wout_missing_data = f_feats_wout_missing_data.append(x)

#input_array = []
#counter = 0
#for x in f_feats_wout_missing_data['DEPARTURE_DELAY']:
#    if x == 1:
#        counter = counter + 1
#        input_array.append([1]) #to categorical?
#    else:
#        counter = counter + 1
#        input_array.append([0])
#        
        
#NEED SPLITTING OF TEST AND TRAINING
        
labels = np.array(f_feats_wout_missing_data['DEPARTURE_DELAY'])
features = f_feats_wout_missing_data.drop(['DEPARTURE_DELAY'], 1)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=1) 


#labels =  pd.DataFrame(input_array)


neigh = KNeighborsClassifier(n_neighbors=25)
neigh.fit(X_train, y_train.ravel()) #need input_array as numpy array
score = neigh.score(X_test, y_test)
print(score)
