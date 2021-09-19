import jax.numpy as np
from jax import vmap

class Dynamics(object):
    def __init__(self, xdim, udim):
        self.xdim = xdim
        self.udim = udim

        self.batchstep = vmap(self.step, in_axes=[0,0,None])
        self.batchstochasticstep = vmap(self.stochasticstep, in_axes=[0,0,0,None])

    def getA(self, xt, ut):
        raise NotImplementedError()

    def getB(self, xt, ut):
        raise NotImplementedError()

    def step(self, xt, ut, dt):
        raise NotImplementedError()

    def stochasticstep(self, xt, ut, dt):
        raise NotImplementedError()

