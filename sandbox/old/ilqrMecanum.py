import numpy as np
import matplotlib.pyplot as plt 
import math

def solveDiscreteTimeVaryingLQR(QList, RList, A, B):
    Klist = []
    Plist = []
    Plist.append(np.zeros((2,2)))
    H = len(QList) 
    for i in range(H):
        K = -np.linalg.inv(RList[H-i-1] + np.transpose (B)@Plist[i]@B)@np.transpose(B)@Plist[i]@A
        Klist.append(K)
        P = QList[H-i-1] + np.transpose(K)@RList[H-i-1]@K + (np.transpose(A+B@K))@Plist[i]@(A+B@K)
        Plist.append(P)
    return Klist 
def solveiLQR(QList, RList, qlist, rlist, A, B, P, p):
    Klist = []
    dlist = []
    H = len(QList) 
    for i in range(H):
        rand = -np.linalg.inv(2*RList[H-i-1] + 2*np.transpose(B)@P@B)
        K = rand@(2*np.transpose(B)@P@A)
        d = rand@(rlist[H-i-1] + B@p)
        Klist.insert(0,K)
        dlist.insert(0,d)
        p = (-2*np.transpose(d)@RList[H-i-1]@K + qlist[H-i-1] - np.transpose(rlist[H-i-1])@K + 2*np.transpose(B@d)@P@(A-B@K)+p@(A-B@K))
        P = QList[H-i-1] + np.transpose(K)@RList[H-i-1]@K + (np.transpose(A+B@K))@P@(A+B@K)
        return Klist, dlist
def solveiLQRv2(QList, RList, qlist, rlist, A, B, P, p):
    K = []
    d = []
    H = len(QList) 
    V_x = p
    V_xx = P
    for i in range(H):
        # Calculating linear approximations of the cost to go function
        Q_x = qlist[H-i-1] + np.dot(A.T, V_x) 
        Q_u = rlist[H-i-1] + np.dot(B.T, V_x)

        #Quadratic Approximations of the cost to go function
        Q_xx = QList[H-i-1] + np.dot(A.T, np.dot(V_xx, A)) 
        #l_ux is zero since there are no cross terms
        # Q_ux = l_ux[t] + np.dot(B.T, np.dot(V_xx, A))
        Q_ux = np.dot(B.T, np.dot(V_xx, A))
        Q_uu = RList[H-i-1] + np.dot(B.T, np.dot(V_xx, B))

        #finding inverse of the second partial with respect to control u
        Q_uu_inv = np.linalg.inv(Q_uu)

        # k = -np.dot(Q_uu^-1, Q_u)
        d.insert(0,-np.dot(Q_uu_inv, Q_u))
        K.insert(0,-np.dot(Q_uu_inv, Q_ux))

        #updating cost to go approximations
        V_x = Q_x - np.dot(K[0].T, np.dot(Q_uu, d[0]))
        V_xx = Q_xx - np.dot(K[0].T, np.dot(Q_uu, K[0]))

    return K,d




def computeHessian(a, b, c, d, x, y):
    cost = computecost(a,b,c,d,x,y)
    H = cost*np.array([[4*(b**2)*((x-c)**2)-2*b, 4*(b**2)*(x-c)*(y-d)],
                       [4*(b**2)*(x-c)*(y-d), 4*(b**2)*((y-d)**2)-2*b]])
    return H

def computeGradient(a, b, c, d, x,  y):
    cost = computecost(a,b,c,d,x,y)
    g = cost*np.array([2*b*(c-x), 2*b*(d-y)])
    return g

def computecost(a, b, c, d, x, y):
    cost = a*math.pow(math.e,-b*((x-c)**2 + (y-d)**2))
    return cost

seconds = 5

x0 = np.array([0,0])

xf = np.array([20,0])

obstacle = np.array([5,0.1])

obstradius = 2

timestep = 0.02

#x(t+1) = A*x(t) + B*u
A = np.eye(2)
B = timestep*np.eye(2)

K = np.eye(2)

trustRatio = 0.99

Q = 1.5*np.eye(2)
R = 1*np.eye(2)
Z = trustRatio*np.eye(2)


Qlis = []
Rlis = []
qlis = []
rlis = []
H = int(seconds/timestep)
for i in range(H):
    Qlis.append(Q)
    Rlis.append(R)
    qlis.append(np.zeros(2))
    rlis.append(np.zeros(2))
    klis, dlis = solveiLQRv2(Qlis,Rlis,qlis,rlis,A,B,Q,np.zeros(2))

# k2, d2 = solveiLQRv2(Qlis,Rlis,qlis,rlis,A,B,10*Q,np.zeros(2))
# print(k2, d2)
# print(klis,dlis)
alpha = 300
beta = 1/((obstradius)**2)

numiterations = 100
xlist = []
u = []


bestx = []
lowestcost = 10000000

j = 0
cost = 0
lastcost = 10000
#creating initial sequence
while(abs(cost-lastcost)>1):
    print("iteration: ", j)
    Qlis = []
    qlis = []
    lastxlist = xlist
    xlist = []
    xt = x0 
    lastcost = cost
    cost = 0
    xlist.append(x0)
    for i in range(H):
        if(j != 0):
            # ut = -klis[i]@(lastxlist[i]-xt) + dlis[i] + u[i]
            ut = -klis[i]@(xf-xt) + dlis[i]
            u[i] = ut
        else:
            ut = -klis[i]@(xf-xt) + dlis[i]
            u.append(ut)

        xt = A@xt + B@ut
        cost += np.transpose(xt)@Q@xt + np.transpose(ut)@R@ut

        if(np.linalg.norm(xt-obstacle)<obstradius+1):
            Qlis.append(Q + computeHessian(alpha,beta,obstacle[0],obstacle[1],xt[0],xt[1]))
            qlis.append(computeGradient(alpha,beta,obstacle[0],obstacle[1],xt[0],xt[1]) - 2*np.transpose(xt)@Q)
            cost += computecost(alpha,beta,obstacle[0],obstacle[1],xt[0],xt[1])
        else:
            Qlis.append(Q)
            qlis.append(np.array([0,0]))
        xlist.append(xt)
    j += 1
    print("total cost: ", cost)
    klis, dlis = solveiLQRv2(Qlis,Rlis,qlis,rlis,A,B,Q,np.zeros(2))
    if(cost<lowestcost):
        lowestcost = cost
        bestx = xlist



fig, ax = plt.subplots()

xlis = []
ylis = []

circle1 = plt.Circle((obstacle[0],obstacle[1]), obstradius , color='r')
ax.add_patch(circle1)
ax.set_aspect('equal', adjustable='datalim')

for state in xlist:
    xlis.append(state[0])
    ylis.append(state[1])

plt.plot(xlis,ylis)
plt.show()















