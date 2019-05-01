#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:30:44 2019

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from sklearn import preprocessing, model_selection
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation

DATA_DIR = 'data'
DATA_FNAME = '2d_5deg_5veh_2obs_euclidean'

HIDDEN_SIZE = 1000
EPOCHS = 5
OPTIMIZER = 'rmsprop'
LOSS = 'mean_squared_error'
MODEL_NAME = f'{OPTIMIZER}_{LOSS}_{EPOCHS}epochs_{str(HIDDEN_SIZE)}'

X_HEADERS = ['x0_i',
             'y0_i',
             'x1_i',
             'y1_i',
             'x2_i',
             'y2_i',
             'x3_i',
             'y3_i',
             'x4_i',
             'y4_i',
             'x0_f',
             'y0_f',
             'x1_f',
             'y1_f',
             'x2_f',
             'y2_f',
             'x3_f',
             'y3_f',
             'x4_f',
             'y4_f',
             'obs0x',
             'obs0y',
             'obs1x',
             'obs1y']




def import_data(directory='.', data_fname='2d_5deg_5veh_2obs_euclidean'):
    # Import the data
    fnames = [i for i in os.listdir(directory) if data_fname in i]
    print('[+] Importing Data...')
    temp = []
    for name in fnames:
        print('  [-] importing ' + name)
        df = pd.read_csv(os.path.join(directory, name))
        temp.append(df)

    data = pd.concat(temp, ignore_index=True)
    print('[+] Imported Data:')
    print(data.head())

    # Properly organize the data
    X = data[X_HEADERS]
    y = data.drop(X_HEADERS, axis=1)
    print('[+] X Data:')
    print(X.head())
    print('[+] Y Data:')
    print(y.head())

    X = np.array(X)
    y = np.array(y)

    return model_selection.train_test_split(X, y, test_size=0.2)


def build_model():
    model = Sequential()
    model.add(Dense(HIDDEN_SIZE, input_dim=24))
    model.add(Activation('relu'))
    model.add(Dense(40, input_dim=HIDDEN_SIZE))
    
    model.compile(optimizer=OPTIMIZER,
                  loss=LOSS,
                  metrics=['accuracy'])
    
    return model


def save_model(model):
    # Increment the number of the file name and save the regressor as a pickle
    print('[+] Saving regressor object...')
    i = 0
    while os.path.exists(MODEL_NAME + '_' + str(i) + '.pickle'):
        i += 1
        
    model.save(MODEL_NAME + '_' + str(i) + '.model')

if __name__ == '__main__':
    # Import data
#    var_names = ['X_train', 'y_train', 'X_test', 'y_test']
    if 'X_train' not in locals():
        X_train, X_test, y_train, y_test = import_data(DATA_DIR, DATA_FNAME)
        X_train = tf.keras.utils.normalize(X_train, axis=1)
        X_test = tf.keras.utils.normalize(X_test, axis=1)
#    # Create the model
#    model = build_model()
#    # Train the model
#    model.fit(X_train, y_train, epochs=EPOCHS)
#    
#    val_loss, val_acc = model.evaluate(X_test, y_test)
#    print(val_loss, val_acc)
#    
#    save_model(model)

    print('[!] Done')