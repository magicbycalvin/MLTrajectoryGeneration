#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:50:53 2019

@author: ckielasjensen
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import scipy.optimize as sop
from tensorflow.keras.models import load_model
import time

import bezier as bez
from optimization import BezOptimization


SAFE_DIST = 1
MODEL = 'rmsprop_mean_squared_error_3epochs_500_0.model'

def generate_random_state(xmin, xmax, ymin, ymax, numPts):
    """
    """
#    pts = [(ub-lb)*np.random.random(2)+lb]
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
    
    model = load_model(MODEL)

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
#                                 maxSpeed=5,
#                                 maxAngRate=1,
                             initPoints=initPoints,
                             finalPoints=finalPoints,
#                             initSpeeds=[1]*numVeh,
#                             finalSpeeds=[1]*numVeh,
#                             initAngs=[np.pi/2]*numVeh,
#                             finalAngs=[np.pi/2]*numVeh,
                             pointObstacles=pointObstacles
                             )

    
    

    ineqCons = [{'type': 'ineq', 'fun': bezopt.temporalSeparationConstraints}]

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
