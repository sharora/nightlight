# import numpy as np
import jax.numpy as np
import math
from . dynamics import Dynamics
from jax import jit, random, vmap
from functools import partial
import time
import numpy as onp

class MecanumDrive(Dynamics):
    def __init__(self):
        Dynamics.__init__(self, 6, 3)
        self._A = np.array([[0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, -0.75, 0, 0],
                            [0, 0, 0, 0, -0.75, 0],
                            [0, 0, 0, 0, 0, -0.75]])
        self._B = np.array([[0,0,0],
                            [0,0,0],
                            [0,0,0],
                            [1,0,0],
                            [0,1,0],
                            [0,0,1]])

    def getA(self, xt, ut):
        return self._A

    def getB(self, xt, ut):
        return self._B

    @partial(jit, static_argnums=(0,))
    def step(self, robotstate, controls, dt):
        '''
        discrete update equation for robot state (x, y theta, xdot, ydot, thetadot)
        very simple double linear integrator model for now, assuming frictional force is
        roughly proportional to velocity
        assumes that we have an input-output coordinate transformation mapping x, y acceleration in the
        global coordinate space to motor torques(we do for bounded control inputs)
        '''
        #simple euler integrator for now
        delta  = (self._A @ robotstate + self._B @ controls) * dt
        robotstate = robotstate + delta
        return robotstate

    @partial(jit, static_argnums=(0,))
    #TODO fix this shite, function is fucked
    def stochasticstep(self, robotstate, controls, dt):
        #original step
        x = self.step(robotstate, controls, dt)

        #adding noise
        vsquared = robotstate[3]**2 + robotstate[4]**2 + robotstate[5]**2
        variance = 0.02*np.clip(np.sqrt(vsquared), 0.000001) #clipping to prevent nan

        #new key for random number generation
        key = random.PRNGKey(0)
        key, subkey = random.split(key)

        disturbance = random.multivariate_normal(subkey, np.zeros(6), variance*np.eye(6))
        return x + disturbance
        # return x

if __name__ == '__main__':
    #testing to see if the jit compiled methods work
    x0 = np.array([72, 36, 0, 0, 0, 0])
    u = np.array([0.5,0.5,0.5])
    d = MecanumDrive()

    d.step(x0, u, 0.02)
    d.stochasticstep(x0, u, 0.02)

    xs = np.stack([x0]*1024)
    us = np.stack([u]*1024)

    #testing batching
    d.batchstep(xs, us, 0.02)

    start = time.time()
    print(d.batchstep(xs, us, 0.02))
    # print(d.stochasticstep(x0, u, 0.02))
    # d.step(x0, u, 0.02)
    print(time.time() - start)


