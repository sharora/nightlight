import numpy as np
import math
from dynamics import Dynamics

class MecanumDrive(Dynamics):
    def __init__(self, xdim, udim):
        Dynamics.__init__(self, xdim, udim)


    def getA(self, xt, ut):
        A = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, -0.5, 0, 0],
                      [0, 0, 0, 0, -0.5, 0],
                      [0, 0, 0, 0, 0, -0.5]])
        return A
    def getB(self, xt, ut):
        B = np.array([[0,0,0],
                      [0,0,0],
                      [0,0,0],
                      [1,0,0],
                      [0,1,0],
                      [0,0,1]])
        return B

    def step(self, robotstate, controls, dt):
        ''' 
        discrete update equation for robot state (x, y theta, xdot, ydot, thetadot)
        very simple double linear integrator model for now, assuming frictional force is
        roughly proportional to velocity
        assumes that we have an input-ouput coordinate transformation mapping x, y acceleratoin in the
        global coordinate space to motor torques(we do for bounded control inputs)
        '''
        A = np.array([[0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, -0.5, 0, 0],
                      [0, 0, 0, 0, -0.5, 0],
                      [0, 0, 0, 0, 0, -0.5]])
        B = np.array([[0,0,0],
                      [0,0,0],
                      [0,0,0],
                      [1,0,0],
                      [0,1,0],
                      [0,0,1]])
        #simple euler integrator for now
        delta  = (A @ robotstate + B @ controls) * dt
        robotstate = robotstate + delta
        return robotstate
    def stochasticstep(self, robotstate, controls, dt):
        vsquared = robotstate[3]**2 + robotstate[4]**2 + robotstate[5]**2
        variance = 0
        if(vsquared > 0.5):
            variance = 0.02*math.sqrt(vsquared)
        x = self.step(robotstate, controls, dt)
        disturbance = np.random.multivariate_normal(np.zeros(6),variance*np.eye(6))
        return x + disturbance

