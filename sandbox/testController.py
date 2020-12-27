from multiprocessing.connection import Client
import random
import numpy as np
from MecanumRobotDynamics import MecanumRobotDynamics

address = ('localhost', 6000)
client = Client(address, authkey=b'Ok Boomer!')

robot = MecanumRobotDynamics(40,100)

x = np.array([110, 20, 45, 0, 0, 0])
k = -10*np.array([[1, 0, 0, 0.5, 0, 0],
             [0, 1, 0, 0, 0.5, 0],
             [0, 0, 1, 0, 0, 0.5]])

xtarg = np.array([72, 72, 90, 0, 0 ,0])

def solveDiscLQR(A, B, Q, R, H):
    P = Q
    Klis = []
    for i in range(H):
        K = -np.linalg.inv(R + np.transpose(B) @ P @ B) @np.transpose(B) @ P @ A
        Klis.insert(0, K)
        P = Q + np.transpose(K) @ R @ K + np.transpose(A + B @ K) @ P @ (A + B @ K)
    return Klis

#in seconds
time = 20
interval = 0.02

#euler integrates derivative
A = robot.getA()*interval + np.eye(6)
B = robot.getB()*interval
H = int(time/interval)
Q = 20*np.eye(6)
R = np.eye(3)

Klis = solveDiscLQR(A, B, Q, R, H)

for i in range(H):
    #testing simple control law
    u = Klis[i]@(x-xtarg)
    x = robot.step(x, u, interval)
    # for watching random walk process
    # client.send([random.randrange(11)-5, random.randrange(11)-5])

    client.send([x[0], x[1], x[2]])

while(True):
    pass

