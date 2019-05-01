#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 09:48:06 2019

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np
import os
#import pandas as pd
import pickle
import scipy.optimize as sop
from tensorflow.keras.models import load_model
import time

import bezier as bez
from optimization import BezOptimization

SAFE_DIST = 1
MODEL = 'rmsprop_mean_squared_error_3epochs_500_0.model'
REG_PATH = 'regressor_distance_k5_0.pickle'
NUM_ITER = 10000


def generate_random_state(xmin, xmax, ymin, ymax, numPts):
    """
    """
    pts = [np.array(((xmax-xmin)*np.random.random()+xmin,
                    (ymax-ymin)*np.random.random()+ymin))]
    while len(pts) < numPts:
        newPt = np.array(((xmax-xmin)*np.random.random()+xmin,
                          (ymax-ymin)*np.random.random()+ymin))
        distances = [np.linalg.norm(newPt-pt) for pt in pts]
        if min(distances) > SAFE_DIST:
            pts.append(newPt)

    return pts


if __name__ == '__main__':

    numVeh = 5
    dim = 2
    deg = 5
    
    # Load the KNN and DNN models
    with open(REG_PATH, 'rb') as f:
        reg = pickle.load(f)
    model = load_model(MODEL)

    stateData = []
    SLData = []
    KNNData = []
    DNNData = []

    for _ in range(NUM_ITER):
        # Generate the random state
        temp = generate_random_state(-10, 10, -10, 10, numVeh*2 + 2)
        initPoints = temp[:numVeh]
        finalPoints = temp[numVeh:-2]
        pointObstacles = temp[-2:]
    
        # Kind of confusing but what it does is unwrap the list of tuples into
        # just a list
        input_state = [j for i in initPoints for j in i]
        input_state += [j for i in finalPoints for j in i]
        input_state += [j for i in pointObstacles for j in i]
    
        bezopt = BezOptimization(numVeh=numVeh,
                                 dimension=dim,
                                 degree=deg,
                                 minimizeGoal='Euclidean',
                                 maxSep=SAFE_DIST,
                                 initPoints=initPoints,
                                 finalPoints=finalPoints,
                                 pointObstacles=pointObstacles
                                 )
        
        stateData.append((input_state, bezopt))
    
        ineqCons = [{'type': 'ineq', 'fun': bezopt.temporalSeparationConstraints}]
    
        # --- Straight Line Opt
        print('starting')
        startTime = time.time()
        xGuess_straightLine = bezopt.generateGuess(std=0)
        results = sop.minimize(
                    bezopt.objectiveFunction,
                    x0=xGuess_straightLine,
                    method='SLSQP',
                    constraints=ineqCons,
                    options={'maxiter': 250,
                             'disp': True,
                             'iprint': 1}
                    )
        endTime = time.time()
    
        print('---')
        print('Straight Line Computation Time: {}'.format(endTime - startTime))
        print('---')
    
        cptsSL = bezopt.reshapeVector(results.x)
        SLData.append((endTime-startTime, results))
        
        # --- KNN Opt
        print('starting')
        startTime = time.time()
        xGuess_KNN = reg.predict(np.atleast_2d(input_state))
        results = sop.minimize(
                    bezopt.objectiveFunction,
                    x0=xGuess_KNN,
                    method='SLSQP',
                    constraints=ineqCons,
                    options={'maxiter': 250,
                             'disp': True,
                             'iprint': 1}
                    )
        endTime = time.time()
    
    
        print('---')
        print('KNN Computation Time: {}'.format(endTime - startTime))
        print('---')
    
        cptsKNN = bezopt.reshapeVector(results.x)
        KNNData.append((endTime-startTime, results))
    
        # --- DNN Opt
        print('starting')
        startTime = time.time()
        xGuess_DNN = model.predict(np.atleast_2d(input_state))
        results = sop.minimize(
                    bezopt.objectiveFunction,
                    x0=xGuess_DNN,
                    method='SLSQP',
                    constraints=ineqCons,
                    options={'maxiter': 250,
                             'disp': True,
                             'iprint': 1}
                    )
        endTime = time.time()
    
        print('---')
        print('DNN Computation Time: {}'.format(endTime - startTime))
        print('---')
    
        cptsDNN = bezopt.reshapeVector(results.x)
        DNNData.append((endTime-startTime, results))
        
    i = 0
    while os.path.exists('state_data_' + str(i) + '.pickle'):
        i += 1
        
    with open('state_data_' + str(i) + '.pickle', 'wb') as f:
        pickle.dump(stateData, f)
        
    with open('SL_data_' + str(i) + '.pickle', 'wb') as f:
        pickle.dump(SLData, f)
        
    with open('KNN_data_' + str(i) + '.pickle', 'wb') as f:
        pickle.dump(KNNData, f)
        
    with open('DNN_data_' + str(i) + '.pickle', 'wb') as f:
        pickle.dump(DNNData, f)

    ###########################################################################
    # Plot Results
    ###########################################################################
    plt.close('all')
    numVeh = bezopt.model['numVeh']
    dim = bezopt.model['dim']
    maxSep = bezopt.model['maxSep']

    ###### Straight Line Results
    fig, ax = plt.subplots()
    curves = []
    for i in range(numVeh):
        curves.append(bez.Bezier(cptsSL[i*dim:(i+1)*dim]))
    for curve in curves:
        plt.plot(curve.curve[0], curve.curve[1], '-',
                 curve.cpts[0], curve.cpts[1], '.--')

    obstacle1 = plt.Circle(bezopt.pointObstacles[0],
                           radius=maxSep,
                           edgecolor='Black',
                           facecolor='red')
    obstacle2 = plt.Circle(bezopt.pointObstacles[1],
                           radius=maxSep,
                           edgecolor='Black',
                           facecolor='green')
    ax.add_artist(obstacle1)
    ax.add_artist(obstacle2)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.title('Straight Line Guess Result', fontsize=28)
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)

    ####### KNN Results
    fig, ax = plt.subplots()
    curves = []
    for i in range(numVeh):
        curves.append(bez.Bezier(cptsDNN[i*dim:(i+1)*dim]))
    for curve in curves:
        plt.plot(curve.curve[0], curve.curve[1], '-',
                 curve.cpts[0], curve.cpts[1], '.--')

    obstacle1 = plt.Circle(bezopt.pointObstacles[0],
                           radius=maxSep,
                           edgecolor='Black',
                           facecolor='red')
    obstacle2 = plt.Circle(bezopt.pointObstacles[1],
                           radius=maxSep,
                           edgecolor='Black',
                           facecolor='green')
    ax.add_artist(obstacle1)
    ax.add_artist(obstacle2)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.title('DNN Guess Result', fontsize=28)
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)

    ###### DNN Initial Guess
    fig, ax = plt.subplots()
    cpts = bezopt.reshapeVector(xGuess_DNN)
    curves = []
    for i in range(numVeh):
        curves.append(bez.Bezier(cpts[i*dim:(i+1)*dim]))
    for curve in curves:
        plt.plot(curve.curve[0], curve.curve[1], '-',
                 curve.cpts[0], curve.cpts[1], '.--')

    obstacle1 = plt.Circle(bezopt.pointObstacles[0],
                           radius=maxSep,
                           edgecolor='Black',
                           facecolor='red')
    obstacle2 = plt.Circle(bezopt.pointObstacles[1],
                           radius=maxSep,
                           edgecolor='Black',
                           facecolor='green')
    ax.add_artist(obstacle1)
    ax.add_artist(obstacle2)
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.title('DNN Initial Guess', fontsize=28)
    plt.xlabel('X Position', fontsize=20)
    plt.ylabel('Y Position', fontsize=20)

    plt.show()
