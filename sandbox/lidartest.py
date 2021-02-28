from multiprocessing.connection import Client
import random
import numpy as np
from MecanumRobotDynamics import MecanumRobotDynamics
from obstacle import CircularObstacle
from particle import Particle
from occupancygrid import OccupancyGrid
from lidarsensor import LidarSensor
import math
from scipy.stats import rv_discrete

address = ('localhost', 6000)
client = Client(address, authkey=b'Ok Boomer!')

#hardcoding occupancy grid
priormap = np.array(
       [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		[0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
		[0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
		[0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
		[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
		[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
		[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
		[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
		[0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
		[0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
		[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

oc = OccupancyGrid(priormap, 12)
ls = LidarSensor(7)

#sending map
client.send(["map", oc])
xt = np.array([72,72, 90, 0, 0, 0])
dt = 0.02


scan = ls.getMeasurement(oc,xt)
client.send(["lidar",  scan])

numberOfParticles = 100
measureskip = 20
particleList = []
for i in range(numberOfParticles):
    xp = np.zeros(6)
    xp[0] = random.random()*144
    xp[1] = random.random()*144
    p = Particle(xp, 1/numberOfParticles)
    particleList.append(p)

# client.send(["points", particleList])

# count = 1
# f= True
# while(f):
#     normalization = 0
#     weightgm = 1
#     weightlist = []
#     indexlist = []
#     z = ls.getMeasurement(oc, xt)
#     for i in range(numberOfParticles):
#         # particleList[i]._x = robot.stochasticstep(particleList[i]._x, u, dt)
#         #calculating new weight: p(zt | x_t_i)

#         if(count % measureskip == 0):
#             xti = particleList[i]._x
#             zi = ls.getMeasurement(oc, xti)
#             newweight = particleList[i]._w*ls.getLaserScanCorrelation(z, zi)
#             particleList[i]._w = newweight
#             normalization += newweight
#     if(count % measureskip == 0):
#         #normalizing the weights
#         for i in range(numberOfParticles):
#             particleList[i]._w = particleList[i]._w / normalization
#             indexlist.append(i)
#             weightlist.append(particleList[i]._w)
#             weightgm *= particleList[i]._w
#     # weightgm = weightgm ** (1/numberOfParticles)
#     # if(weightgm < 0.1*(1/numberOfParticles)):
#     #     #resample particles lel
#     #     sample=rv_discrete(values=(indexlist,weightlist)).rvs(size=numberOfParticles)
#     #     newplist = []
#     #     for i in range(numberOfParticles):
#     #         p = particleList[sample[i]]
#     #         newplist.append(Particle(p._x, 1/numberOfParticles))
#     #     particleList = newplist
#     client.send(["points", particleList])
#     client.send(xt)
#     count += 1
#     if(count == 1):
#         f = True
while(True):
    pass
