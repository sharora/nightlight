import numpy as np
from scipy.stats import rv_discrete
from . particle import Particle

class ParticleFilter(object):
    def __init__(self, particleList, dynamics, sensor, k_neff):
        self._numberOfParticles = len(particleList)
        self._particleList = particleList
        self._dynamics = dynamics
        self._sensor = sensor
        self._k_neff = k_neff
    def step(self, u, dt):
        #for each particle, use the given dynamics function to update its state
        for particle in self._particleList:
            particle._x = self._dynamics.stochasticstep(particle._x, u, dt)
    def updateweights(self, z, oc = None):
        #oc is optional parameter: occupancy grid
        #n is some normalization to make everything sum to one
        n = 0

        #for each particle, update its weight using bayes rule: p(x|z) =
        #p(z|x)p(x)*n
        for particle in self._particleList:
            if(oc == None):
                particle._w *= self._sensor.getMeasurementProbability(particle._x, z)
            else:
                particle._w *= self._sensor.getMeasurementProbability(particle._x, z, oc)
            n += particle._w
        #calculating the number of effective particles in the same loop to determine
        #if resampling is needed
        neff = 0
        #normalization of all weights
        for particle in self._particleList:
            particle._w = particle._w / n
            neff += (particle._w)**2
        neff = 1.0/neff

        #resampling if needed
        if(neff < self._k_neff * self._numberOfParticles):
            self.resampleParticles()

    def resampleParticles(self):
        indexlist = []
        weightlist = []
        for i in range(self._numberOfParticles):
            indexlist.append(i)
            weightlist.append(self._particleList[i]._w)

        #resampling
        sample=rv_discrete(values=(indexlist,weightlist)).rvs(size=self._numberOfParticles)
        newplist = []
        for i in range(self._numberOfParticles):
            p = self._particleList[sample[i]]
            newplist.append(Particle(p._x, 1/self._numberOfParticles))
        self._particleList = newplist
    def getMaxParticle(self):
        maxweight = 0
        maxparticle = None
        for particle in self._particleList:
            if(particle._w > maxweight):
                maxweight = particle._w
                maxparticle = particle
        return maxparticle


