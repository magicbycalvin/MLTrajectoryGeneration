#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
import pandas as pd
import scipy.optimize as sop
import time

import bezier as bez
from optimization import BezOptimization

SAFE_DIST = 1
NUM_RUNS = 100000
PLOT = False
SAVE_NAME = '2d_5deg_5veh_2obs_euclidean_11.csv'


def animateTrajectory(trajectories):
    """Animates the trajectories

    """
    global ani

    curveLen = len(trajectories[0].curve[0])
    fig, ax = plt.subplots()
    [ax.plot(traj.curve[0], traj.curve[1], '-', lw=3) for traj in trajectories]
    lines = [ax.plot([], [], 'o', markersize=20)[0] for traj in trajectories]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(frame):
        for i, line in enumerate(lines):
            traj = trajectories[i]
            try:
                line.set_data(traj.curve[0][frame],
                              traj.curve[1][frame])
            except IndexError:
                line.set_data(traj.curve[0][curveLen-frame-1],
                              traj.curve[1][curveLen-frame-1])
        return lines

    plt.axis('off')
    ani = animation.FuncAnimation(fig,
                                  animate,
                                  len(trajectories[0].curve[0])*2,
                                  init_func=init,
                                  interval=10,
                                  blit=True,
                                  repeat=True)

    plt.show()


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
    plt.close('all')

    numVeh = 5
    dim = 2
    deg = 5

    # Create the dataframe that will hold the desired data
    column_headers = ['x0_i',
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
    results_headers = ['res'+str(i) for i in range(numVeh*dim*(deg-1))]
    column_headers += results_headers
    dataToSave = pd.DataFrame(columns=column_headers)
    # Initialize the file if it doesn't exist
    if not os.path.exists(SAVE_NAME):
        with open(SAVE_NAME, 'w') as f:
            dataToSave.to_csv(f, index=False)

    for _ in range(NUM_RUNS):
        print('Current Iteration ===> {}'.format(_))
        temp = generate_random_state(-10, 10, -10, 10, numVeh*2 + 2)
        initPoints = temp[:numVeh]
        finalPoints = temp[numVeh:-2]
        pointObstacles = temp[-2:]

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

        xGuess = bezopt.generateGuess(std=0)
        ineqCons = [{'type': 'ineq', 'fun': bezopt.temporalSeparationConstraints}]
#                    {'type': 'ineq', 'fun': bezopt.maxSpeedConstraints},
#                    {'type': 'ineq', 'fun': bezopt.maxAngularRateConstraints},
#                    {'type': 'ineq', 'fun': lambda x: x[-1]}]

#        _ = bez.Bezier(bezopt.reshapeVector(xGuess))
#        _.elev(10)
#        _ = _*_

        startTime = time.time()
        print('starting')
        results = sop.minimize(
                    bezopt.objectiveFunction,
                    x0=xGuess,
                    method='SLSQP',
                    constraints=ineqCons,
                    options={'maxiter': 250,
                             'disp': True,
                             'iprint': 1}
                    )
        endTime = time.time()

        count = 0
        while not results.success:
            if count > 10:
                break
            xGuess = bezopt.generateGuess(std=1+count/3)
            startTime = time.time()
            print('starting again')
            results = sop.minimize(
                        bezopt.objectiveFunction,
                        x0=xGuess,
                        method='SLSQP',
                        constraints=ineqCons,
                        options={'maxiter': 250,
                                 'disp': True,
                                 'iprint': 2}
                        )
            endTime = time.time()
            count += 1

        print('---')
        print('Computation Time: {}'.format(endTime - startTime))
        print('---')

        cpts = bezopt.reshapeVector(results.x)

        ###
        # Save the results
        ###

        # Kind of confusing but what it does is unwrap the list of tuples into
        # just a list
        data = [j for i in initPoints for j in i]
        data += [j for i in finalPoints for j in i]
        data += [j for i in pointObstacles for j in i]

        data += list(results.x)

        dataToSave.loc[0] = data

        with open(SAVE_NAME, 'a') as f:
            dataToSave.to_csv(f, index=False, header=False)

        ###########################################################################
        # Plot Results
        ###########################################################################
        if PLOT:
            plt.close('all')
            numVeh = bezopt.model['numVeh']
            dim = bezopt.model['dim']
            maxSep = bezopt.model['maxSep']

            fig, ax = plt.subplots()
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
            plt.title('Vehicle Trajectory', fontsize=28)
            plt.xlabel('X Position', fontsize=20)
            plt.ylabel('Y Position', fontsize=20)

            animateTrajectory(curves)
