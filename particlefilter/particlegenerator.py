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
    particleList = []
    for i in range(numparticles):
        p = Particle(np.random.uniform(boxmin, boxmax), 1.0/numparticles)
        particleList.append(p)

    return particleList

