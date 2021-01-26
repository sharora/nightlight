from multiprocessing.connection import Client
import random
import numpy as np
from MecanumRobotDynamics import MecanumRobotDynamics
from obstacle import CircularObstacle
from particle import Particle
import math

address = ('localhost', 6000)
client = Client(address, authkey=b'Ok Boomer!')

robot = MecanumRobotDynamics(40,100)

#initializing start and target states
x0 = np.array([72, 72, 90, 20, 20, 0])

#in seconds
# time = 3
dt = 0.02

def sampleMeasurement(mu, sigma):
    return np.random.multivariate_normal(mu, sigma*np.eye(6))

def gaussianPDF(mu, point, sigma):
    cov = sigma*np.eye(6)
    constant = 1/((2*math.pi)**(3) * np.linalg.det(cov)**(1/2))
    exponent = -1/2 * ((point - mu).T @ np.linalg.inv(cov) @ (point - mu))
    return constant*math.exp(exponent)


numberOfParticles = 100
measureskip = 50

particleList = []
for i in range(numberOfParticles):
    # p = Particle(x0 + np.random.normal(0,5,size=[6,]), 1/numberOfParticles)
    p = Particle(x0, 1/numberOfParticles)
    particleList.append(p)

client.send(["points", particleList])

xt = x0
count = 0
while(True):
    normalization = 0
    xt = robot.step(xt, np.zeros(3), dt)
    z = sampleMeasurement(xt, 10)
    for i in range(numberOfParticles):
        particleList[i]._x = robot.stochasticstep(particleList[i]._x, np.zeros(3), dt)
        #calculating new weight: p(zt | x_t_i)

        if(count % measureskip == 0):
            xti = particleList[i]._x
            newweight = particleList[i]._w*(gaussianPDF(xti, z, 10))
            particleList[i]._w = newweight
            normalization += newweight
    if(count % measureskip == 0):
        #normalizing the weights
        for i in range(numberOfParticles):
            particleList[i]._w = particleList[i]._w / normalization
    client.send(["points", particleList])
    client.send(xt)
    count += 1


