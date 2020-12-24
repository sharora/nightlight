from multiprocessing.connection import Client
import random
import numpy as np
from MecanumRobotDynamics import MecanumRobotDynamics

address = ('localhost', 6000)
client = Client(address, authkey=b'Ok Boomer!')

robot = MecanumRobotDynamics(40,100)

x = np.array([110, 20, 45, 0, 0, 0])
k = 3*np.array([[1, 0, 0, 1, 0, 0],
             [0, 1, 0, 0, 1, 0],
             [0, 0, 1, 0, 0, 1]])

xtarg = np.array([72, 72, 90, 0, 0 ,0])
while(True):
    #testing simple control law
    u = k@(xtarg - x)

    x = robot.step(x, u, 0.02)
    # for watching random walk process
    # client.send([random.randrange(11)-5, random.randrange(11)-5])

    client.send([x[0], x[1], x[2]])

