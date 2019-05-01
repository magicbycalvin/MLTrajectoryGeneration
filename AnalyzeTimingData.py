#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:20:21 2019

@author: ckielasjensen
"""
import numpy as np
#    stateData(input_state, bezopt)
#    SLData((endTime-startTime, results))
#    KNNData((endTime-startTime, results))
#    DNNData((endTime-startTime, results))

SLTimes = np.array([i[0] for i in SLData])
KNNTimes = np.array([i[0] for i in KNNData])
DNNTimes = np.array([i[0] for i in DNNData])

allTimes = [SLTimes, KNNTimes, DNNTimes]
strTimes = ['Straight Line Guess', 'KNN Guess', 'DNN Guess']

print(f'Length of the test data: {len(SLTimes)}')

print(' MIN  | MEAN  |  MAX   |  STD')
for i, t in enumerate(allTimes):
    msg = (f'{t.min():.3f} | {t.mean():.3f} | {t.max():.3f} | {t.std():.3f}'
           f' --- {strTimes[i]}')
    print(msg)
    
BestKNN = [val-KNNTimes[i] for i, val in enumerate(SLTimes) if val > KNNTimes[i]]
BestDNN = [val-DNNTimes[i] for i, val in enumerate(SLTimes) if val > DNNTimes[i]]

print(f'Number of times KNN was better: {len(BestKNN)}')
print(f'Average better time: {np.mean(BestKNN)}')

print(f'Number of times DNN was better: {len(BestDNN)}')
print(f'Average better time: {np.mean(BestDNN)}')