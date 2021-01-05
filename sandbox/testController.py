from multiprocessing.connection import Client
import random
import numpy as np
from MecanumRobotDynamics import MecanumRobotDynamics
from obstacle import CircularObstacle

address = ('localhost', 6000)
client = Client(address, authkey=b'Ok Boomer!')

robot = MecanumRobotDynamics(40,100)

#initializing start and target states
x0 = np.array([12, 71, 90, 0, 0, 0])
xtarg = np.array([130, 71, 90, 0, 0 ,0])

#initializing obstacle and sending info to visualizer
obst1 = CircularObstacle(72, 72, 10, 5)
client.send(["obstacle", [obst1]])

#manual feedback controller
# k = -10*np.array([[1, 0, 0, 0.5, 0, 0],
#              [0, 1, 0, 0, 0.5, 0],
#              [0, 0, 1, 0, 0, 0.5]])

def solveDiscLQR(A, B, Q, R, H):
    P = Q
    Klis = []
    for i in range(H):
        K = -np.linalg.inv(R + np.transpose(B) @ P @ B) @np.transpose(B) @ P @ A
        Klis.insert(0, K)
        P = Q + np.transpose(K) @ R @ K + np.transpose(A + B @ K) @ P @ (A + B @ K)
    return Klis
def solveiLQR(QList, RList, qlist, rlist, A, B, P, p, C):
    '''
    Assuming linear dynamics for now, will add time varying(for nonlinear approx) later
    '''
    Klist = []
    dlist = []
    H = len(QList) 
    for i in range(H):
        #inverseterm is not negated because I am using -Kx notation for control law
        inverseterm = np.linalg.inv(2*RList[H-i-1] + np.transpose(p)@B + 2*np.transpose(B)@P@B)
        K = inverseterm@(2*np.transpose(B)@P@A)
        d = inverseterm@(rlist[H-i-1] + 2*np.transpose(C)@P@B)
        Klist.insert(0,K)
        dlist.insert(0,d)
        p = (qlist[H-i-1] - np.transpose(rlist[H-i-1])@K - 2*np.transpose(d)@RList[H-i-1]@K + p@(A-B@K) + 2*np.transpose(B@d)@P@(A-B@K) + 2*np.transpose(C)@P@(A-B@K))
        P = (np.transpose(A-B@K))@P@(A-B@K) + QList[H-i-1] + np.transpose(K)@RList[H-i-1]@K 
    return Klist, dlist

#in seconds
time = 3
interval = 0.02

#euler integrates derivative
A = robot.getA()*interval + np.eye(6)
B = robot.getB()*interval
H = int(time/interval)
Q = 20*np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0.5, 0, 0],
              [0, 0, 0, 0, 0.5, 0],
              [0, 0, 0, 0, 0, 0.5]])
R = np.eye(3)
x = x0

#initializing lists for cost function approximations, state, and controls
Qlis = []
Rlis = []
qlis = []
rlis = []
ulis = []
xlis = []
lastxlis = []
for i in range(H):
    Rlis.append(R)
    rlis.append(np.zeros(3))

#initializing the first guess for the trajectory
klis = solveDiscLQR(A, B, Q, R, H)
totalcost = 0
lastcost = 0

#sampling trajectory to initialize lists
for i in range(H):
    #testing simple control law
    u = klis[i]@(x-xtarg)
    x = robot.step(x, u, interval)
    ulis.append(u)
    xlis.append(x)
    totalcost += obst1.getCost(x, 9)
    # hess = obst1.getCostHessian(x, 9)
    # if(not np.all(np.linalg.eigvals(hess) >= 0)):
    #     print(totalcost)
    client.send([x[0], x[1], x[2]])

#ilqr optimization
j = 0
while(abs(totalcost-lastcost)>1):
    print("iteration: ", j)
    Qlis = []
    qlis = []
    lastxlis = xlis
    xlis = []
    x = x0
    lastcost = totalcost
    totalcost = 0
    xlis.append(x0)
    for i in range(H):
        if(j == 0):
            u = klis[i]@(x-xtarg)
        else:
            u = -klis[i]@(x - xtarg) + dlis[i]

        ulis.append(u)
        x = A@x + B@u
        totalcost += np.transpose(x)@Q@x + np.transpose(u)@R@u + obst1.getCost(x,9)

        Qt = obst1.getCostHessian(x, 9) + Q
        qt = obst1.getCostGradient(x, 9) - 2*np.transpose(x)@Q
        Qlis.append(Qt)
        qlis.append(qt)

        xlis.append(x)
        client.send([x[0], x[1], x[2]])
    j += 1
    print("total cost: ", totalcost)
    klis, dlis = solveiLQR(Qlis,Rlis,qlis,rlis,A,B,Q,np.zeros(6),np.zeros(6))
