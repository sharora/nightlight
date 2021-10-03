import jax.numpy as np
import numpy as onp
from jax import grad, jit
import math
import matplotlib.pyplot as plt
from visualizer.visualizer import Visualizer
from trajplanners.sgdplanner import GradientDescentPlanner
from trajplanners.ilqrplanner import iLQRPlanner
from dynamics.mecanumdrive import MecanumDrive
from costfunctions.mecanumobstaclecost import MecanumObstacleCost
import time


'''Configuration Starts'''
#initializing various constants for optimization loop
#some are defined by problem, some need tuning

x0 = np.array([10., 70., 0., 0., 0., 0.])
x_targ = np.array([100., 80., 0., 0., 0., 0.])
obstacles = np.array([[72., 100., 1.]]) #, [72., 40., 20.], [72., 90., 20.]])

robot_length = 18
robot_radius = robot_length/2
dt = 0.1
T = 4
h = int(T/dt)
u_0 = np.array([1., 1., 0.])
u = np.vstack([u_0]*h)

steps = 100
lr = 0.01

Q = 1*np.array([
    [1., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.]
    ])
Qf = 10*np.array([
    [1., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0.1, 0., 0.],
    [0., 0., 0., 0., 0.1, 0.],
    [0., 0., 0., 0., 0., 0.1]
    ])
R = 0.01*np.array([
    [1.0, 0., 0.],
    [0., 1.0, 0.],
    [0., 0., 1.0]
])

'''Configuration Ends'''

dynamics = MecanumDrive()
costfunction = MecanumObstacleCost(robot_radius, obstacles, Q, R, Qf)
ilqr_planner = iLQRPlanner(dynamics, costfunction, h, dt, lr)

start = time.time()
u = ilqr_planner.optimizeTrajectory(u, x0, x_targ, steps)
print(time.time() - start)

#computing final trajectory using controls and displaying it
viz = Visualizer(144, 144, 6)

x_t = x0
obst = onp.array(obstacles)
for i in range(h):
    #step
    x_t = dynamics.step(x_t, u[i], dt)

    #converting to numpy array
    xg = onp.array(x_t)

    #graph all objects
    for i in range(obst.shape[0]):
        viz.graphCircularObstacle(obst[i][0], obst[i][1], obst[i][2])

    viz.graphSquareRobot(xg[0], xg[1], xg[2],
                         robot_length)
    viz.updateFrame()
    time.sleep(0.05)
    viz.clearScreen((0,0,0))
