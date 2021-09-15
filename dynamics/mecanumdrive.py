import numpy as np
import math
from . dynamics import Dynamics

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

    def stochasticstep(self, robotstate, controls, dt):
        vsquared = robotstate[3]**2 + robotstate[4]**2 + robotstate[5]**2
        variance = 0
        x = self.step(robotstate, controls, dt)
        disturbance = 0
        if(vsquared > 0.5):
            variance = 0.02*math.sqrt(vsquared)
            disturbance = np.random.multivariate_normal(np.zeros(6),variance*np.eye(6))
        return x + disturbance

