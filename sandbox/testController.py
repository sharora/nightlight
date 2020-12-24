from multiprocessing.connection import Client
import random
import numpy as np
from MecanumRobotDynamics import MecanumRobotDynamics

address = ('localhost', 6000)
client = Client(address, authkey=b'Ok Boomer!')

robot = MecanumRobotDynamics(40,100)

x = np.array([72, 72, 90, 25, 0, 10])


while(True):
    x = robot.step(x, np.array([0,0,0]), 0.02)
    # for watching random walk process
    # client.send([random.randrange(11)-5, random.randrange(11)-5])

    client.send([x[0], x[1], x[2]])

