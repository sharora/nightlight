import numpy as np
from scipy.stats import rv_discrete
from . particle import Particle
import time

class ParticleFilter(object):
    def __init__(self, states, weights, dynamics, sensor, k_neff):
        self._states = states
        self._weights = weights
        self._numberOfParticles = states.shape[0]
        self._dynamics = dynamics
        self._sensor = sensor
        self._k_neff = k_neff

    def step(self, u, dt):
        #for each particle, use the given dynamics function to update its state
        for i in range(self._numberOfParticles):
            self._states[i]= self._dynamics.stochasticstep(
                self._states[i], u, dt)

    def updateweights(self, z, oc = None):
        #oc is optional parameter: occupancy grid
        #n is some normalization to make everything sum to one
        n = 0

        #for each particle, update its weight using bayes rule: p(x|z) =
        #p(z|x)p(x)*n
        measureprobs = np.zeros(self._weights.shape)
        for i in range(self._numberOfParticles):
            if(oc == None):
                measureprobs[i] = self._sensor.getMeasurementProbability(self._states[i], z)
            else:
                measureprobs[i] = self._sensor.getMeasurementProbability(self._states[i], z, oc)

        #exponentiation of probs (scores) to increase inequality
        measureprobs = np.exp(measureprobs - np.max(measureprobs))

        #multiplying old probabilities by measurement probabilities
        self._weights = np.multiply(self._weights, measureprobs)

        #normalization
        self._weights = self._weights/self._weights.sum()

        #calculating neff
        neff = np.sum(self._weights**2)
        neff = 1.0/neff

        #resampling if needed
        if(neff < self._k_neff * self._numberOfParticles):
            self.resampleParticles()

    def resampleParticles(self):
        indexarr = np.arange(0, self._numberOfParticles)

        #resampling
        sample=rv_discrete(values=(indexarr,
                                   self._weights)).rvs(size=self._numberOfParticles)

        self._states = self._states[sample]
        self._weights = 1.0/self._numberOfParticles * np.ones((self._numberOfParticles,))

    def getMaxParticle(self):
        p = np.argmax(self._weights)
        return self._states[p], self._weights[p]

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


