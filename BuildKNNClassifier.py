#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os

import numpy as np
import pickle
import pandas as pd
from sklearn import model_selection, preprocessing
from sklearn.neighbors import KNeighborsRegressor

DATA_FNAME = '2d_5deg_5veh_2obs_euclidean'
PICKLE_NAME = 'regressor_distance_k5'
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

    return X, y


if __name__ == '__main__':
#    # Import the data
#    fnames = [i for i in os.listdir() if DATA_FNAME in i]
#    print('[+] Importing Data...')
#    temp = []
#    for name in fnames:
#        print('  [-] importing ' + name)
#        df = pd.read_csv(name)
#        temp.append(df)
#
#    data = pd.concat(temp, ignore_index=True)
#    print('[+] Imported Data:')
#    print(data.head())
#
#    # Properly organize the data
#    X = data[X_HEADERS]
#    y = data.drop(X_HEADERS, axis=1)
#    print('[+] X Data:')
#    print(X.head())
#    print('[+] Y Data:')
#    print(y.head())
#
#    X = np.array(X)
#    y = np.array(y)
#
#    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    X, y = import('data')
    X_train, X_test_val, y_train, y_test_val = model_selection.train_test_split(X, y, test_size=0.2)
    X_test, X_validation, y_test, y_validation = model_selection.train_test_split(X_test_val, y_test_val, test_size=0.5)
    # Train the regressor
    reg = KNeighborsRegressor(n_neighbors=5, n_jobs=-1, weights='distance')
    print('[+] Fitting...')
    reg.fit(X_train, y_train)
    print('  [-] Fit complete')

#    print('[+] Scoring...')
#    acc = reg.score(X_test, y_test)
#    print('  [-] R^2 Score: {:0.2f}'.format(acc))
#
#    # Increment the number of the file name and save the regressor as a pickle
#    print('[+] Saving regressor object...')
#    i = 0
#    while os.path.exists(PICKLE_NAME + '_' + str(i) + '.pickle'):
#        i += 1
#    with open(PICKLE_NAME + '_' + str(i) + '.pickle', 'wb') as f:
#        pickle.dump(reg, f)

    print('[!] Done')
