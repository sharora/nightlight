import jax.numpy as np
from jax import grad
from jax import jit
import math
import matplotlib.pyplot as plt 


def computecost(x_t, obst_state, obst_radius):
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
    c = (np.transpose(x_trans) @ X @ x_trans)[0][0]

    if(c < (obst_radius**2)):
        c = c / ((2*obst_radius)**2)
        return (obst_radius**4) * np.exp(-c)
    else:
        return 0.0

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

def simulation_loop(x0, x_targ, ulis, obst_state, obst_radius, dt, steps, Q, R):
    totalcost = 0
    x_t = x0
    for i in range(steps):
        #forward dynamics (applying control)
        u_t = ulis[i]
        x_t = step(x_t, u_t, dt)

        #adding to cost
        err = x_t - x_targ
        totalcost += (np.transpose(err) @ Q @ err)[0][0]
        totalcost += (np.transpose(u_t)@ R @ u_t)[0][0]
        totalcost += 100*computecost(x_t, obst_state, obst_radius)
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

Q = 10*np.array([
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

#initializing jit function
# ulis_grad = jit(grad(simulation_loop, argnums=2))
# ulis_grad(x0, x_targ, ulis, obst_state, obst_radius, dt, steps, Q, R)

numsteps = 100
lr = 0.01

#main optimization loop
for i in range(numsteps):
    # ulis_g = ulis_grad(x0, x_targ, ulis, obst_state, obst_radius, dt, steps, Q, R)
    ulis_g = grad(simulation_loop, argnums=2)(x0, x_targ, ulis, obst_state, obst_radius, dt, steps, Q, R)
    #updating the weights
    for j in range(steps):
        ulis[j] -= lr*ulis_g[j]
    #showing current cost
    if(i % 10 == 0):
        print(simulation_loop(x0, x_targ, ulis, obst_state, obst_radius, dt, steps, Q, R))



#simple function tests
# print(step(x0, u_0, dt))
# print(computecost(x0, obst_state, obst_radius))
# print(grad(computecost, argnums=1)(x0, obst_state, obst_radius))
# print(simulation_loop(x0, x_targ, ulis, obst_state, obst_radius, dt, steps, Q, R))


#computing final trajectory using controls
xlist = []
x_t = x0
for i in range(steps):
    xlist.append(x_t)
    x_t = step(x_t, ulis[i], dt)

#showing final trajectory
fig, ax = plt.subplots()

xlis = []
ylis = []

circle1 = plt.Circle((obst_state[0][0], obst_state[1][0]), obst_radius , color='r')
ax.add_patch(circle1)
ax.set_aspect('equal', adjustable='datalim')

for state in xlist:
    xlis.append(state[0])
    ylis.append(state[1])

plt.plot(xlis,ylis)
plt.show()




