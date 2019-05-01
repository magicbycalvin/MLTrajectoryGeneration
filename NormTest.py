#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:54:02 2019

@author: ckielasjensen
"""

from numba import njit
import numpy as np


@njit
def norm(x):
    summation = 0
    for val in x:
        summation += val*val
        
    return np.sqrt(summation)


summation = 0
for i in range(len(X_test)):
    print(i)
    pred = reg.predict(np.atleast_2d(X_test[i]))[0]
    summation += norm(pred-y_test[i])