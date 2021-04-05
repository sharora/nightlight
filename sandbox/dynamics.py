import numpy as np

class Dynamics(object):
    def __init__(self, xdim, udim):
        self.xdim = xdim
        self.udim = udim

    def getA(self, xt, ut):
        raise NotImplementedError()

    def getB(self, xt, ut):
        raise NotImplementedError()

    def step(self, xt, ut, dt):
        raise NotImplementedError()

    def stochasticstep(self, xt, ut, dt):
        raise NotImplementedError()

