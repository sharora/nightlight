from multiprocessing.connection import Client
import random
import numpy as np
from mecanumdrive import MecanumDrive
from obstacle import CircularObstacle
from particle import Particle
import math
from scipy.stats import rv_discrete

address = ('localhost', 6000)
client = Client(address, authkey=b'Ok Boomer!')

robot = MecanumDrive(6,3)

#initializing start and target states
x0 = np.array([71, 71, 90, 0, 0, 0])
xtarg = np.array([130, 71, 90, 0, 0 ,0])
k = 5*np.array([[1, 0, 0, 0.5, 0, 0],
             [0, 1, 0, 0, 0.5, 0],
             [0, 0, 1, 0, 0, 0.5]])

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
measureskip = 10

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
    weightgm = 1
    weightlist = []
    indexlist = []
    u = -k @ (xt - xtarg)
    xt = robot.step(xt, u, dt)
    z = sampleMeasurement(xt, 10)
    for i in range(numberOfParticles):
        particleList[i]._x = robot.stochasticstep(particleList[i]._x, u, dt)
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
            indexlist.append(i)
            weightlist.append(particleList[i]._w)
            weightgm *= particleList[i]._w
    weightgm = weightgm ** (1/numberOfParticles)
    if(weightgm < 0.1*(1/numberOfParticles)):
        #resample particles lel
        sample=rv_discrete(values=(indexlist,weightlist)).rvs(size=numberOfParticles)
        newplist = []
        for i in range(numberOfParticles):
            p = particleList[sample[i]]
            newplist.append(Particle(p._x, 1/numberOfParticles))
        particleList = newplist
    client.send(["points", particleList])
    client.send(xt)
    count += 1
    if(count == 300):
        xtarg = np.array([random.randint(10,130),random.randint(10,140),0,0,0,0])
        count = 0




