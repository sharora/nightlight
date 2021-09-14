import jax.numpy as np
from jax import grad
from jax import jit
import math
import matplotlib.pyplot as plt
from visualizer.visualizer import Visualizer
import time


@jit
def computecost(x_t, obst_state, obst_radius, robot_radius):
    #finding the distance between the states
    x_trans = x_t - obst_state
    X = np.array([
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]
        ])
    obst_radius += robot_radius
    dist = (np.transpose(x_trans) @ X @ x_trans)[0][0]
    indicator =  np.heaviside(obst_radius**2 - dist, 0.5)

    c = dist / ((2*obst_radius)**2)
    cost = (obst_radius**4) * np.exp(-c) * indicator
    return cost

@jit
def step(x_t, u_t, dt):
    #assuming mecanum dynamics for now
    A = np.array([[0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 0., 1.],
                    [0., 0., 0., -0.5, 0., 0.],
                    [0., 0., 0., 0., -0.5, 0.],
                    [0., 0., 0., 0., 0., -0.5]])
    B = np.array([[0.,0.,0.],
                    [0.,0.,0.],
                    [0.,0.,0.],
                    [1.,0.,0.],
                    [0.,1.,0.],
                    [0.,0.,1.]])
    delta  = (A @ x_t + B @ u_t) * dt
    x_next = x_t + delta
    return x_next

@jit
def simulation_loop(x0, x_targ, ulis, obst_state, obst_radius, robot_radius, dt, Q, R):
    totalcost = 0
    x_t = x0
    for i in range(20):
        #forward dynamics (applying control)
        u_t = ulis[i]
        x_t = step(x_t, u_t, dt)

        #adding to cost
        err = x_t - x_targ
        totalcost += (np.transpose(err) @ Q @ err)[0][0]
        totalcost += (np.transpose(u_t)@ R @ u_t)[0][0]
        totalcost += 100*computecost(x_t, obst_state, obst_radius, robot_radius)
    return totalcost

#initializing various constants for optimization loop
#some are defined by problem, some need tuning
x0 = np.array([
    [10.],
    [70.],
    [0.],
    [0.],
    [0.],
    [0.]
])
x_targ = np.array([
    [100.],
    [80.],
    [0.],
    [0.],
    [0.],
    [0.]
])
obst_state = np.array([
    [50.],
    [80.],
    [0.],
    [0.],
    [0.],
    [0.]
])
obst_radius = 10
robot_length = 18
dt = 0.1
T = 2
steps = int(T/dt)
ulis = []
for i in range(steps):
    u_0 = np.array([
        [1.],
        [1.],
        [0.]
    ])
    ulis.append(u_0)

ulis = np.array(ulis)

Q = 25*np.array([
    [1., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0.],
    [0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0.]
    ])
R = 1*np.array([
    [1.0, 0., 0.],
    [0., 1.0, 0.],
    [0., 0., 1.0]
])

#initializing jit function for calculating gradients
ulis_grad = jit(grad(simulation_loop, argnums=2))

#running all jit functions once so they can be compiled/traced
computecost(x0, obst_state, obst_radius, robot_length/2)
step(x0, ulis[0], dt)
simulation_loop(x0, x_targ, ulis, obst_state, obst_radius, robot_length/2, dt, Q, R)
ulis_grad(x0, x_targ, ulis, obst_state, obst_radius, robot_length/2, dt, Q, R)

numsteps = 1000
lr = 0.001

start = time.time()
#main optimization loop
for i in range(numsteps):
    ulis_g = ulis_grad(x0, x_targ, ulis, obst_state, obst_radius, robot_length/2,
                       dt, Q, R)

    #updating the weights
    ulis -= lr*ulis_g
    #showing current cost
    # if(i % 10 == 0):
    #     print(simulation_loop(x0, x_targ, ulis, obst_state, obst_radius, dt,
    #                           Q, R))

print(time.time() - start)


#computing final trajectory using controls and displaying it
viz = Visualizer(144, 144, 6)

x_t = x0
for i in range(steps):
    x_t = step(x_t, ulis[i], dt)
    viz.clearScreen((0,0,0))
    viz.graphCircularObstacle(float(obst_state[0][0]), float(obst_state[1][0]),
                              obst_radius)
    viz.graphSquareRobot(float(x_t[0][0]), float(x_t[1][0]), float(x_t[2][0]),
                         robot_length)
    viz.updateFrame()
    time.sleep(0.05)

