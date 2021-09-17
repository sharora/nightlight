import numpy as np
from . particle import Particle

def randomParticlesBox(numparticles, boxmin, boxmax):
    """Generates particles from a box in R^n

    Args:
       numparticles : number of particles states to gen
       boxmin : an array of box minimums in each axis
       boxmax : an array of box maximums in each axis

    Returns:
        An array of particles states
    """
    states = np.zeros((numparticles, boxmin.shape[0]))
    weights = np.zeros((numparticles,))

    for i in range(numparticles):
        states[i] = np.random.uniform(boxmin, boxmax)
        weights[i] = 1.0/numparticles

    return states, weights

def uniformParticles(numparticles, state):
    """Returns particles which all share the same state as the given state.

    Args:
       numparticles : number of particles states to gen
       state : an np array describing the state

    Returns:
       An array of particle states
    """
    states = np.zeros((numparticles, state.shape[0]))
    weights = np.zeros((numparticles,))

    for i in range(numparticles):
        states[i] = state
        weights[i] = 1.0/numparticles

    return states, weights


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
