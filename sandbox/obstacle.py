import numpy as np
import math

class CircularObstacle(object):
    def __init__(self, x, y, radius, penalty):
        super().__init__()
        self._x = x
        self._y = y
        self._radius = radius
        self._penalty = penalty

    def getCost(self, robotstate, robotradius):
        #TODO make cost function incorporate radius
        #add cost function that is inversely proportional to distance
        # clips to zero, when robot is outside radius
        x = self._x - robotstate[0]
        y = self._y - robotstate[1]

        d = math.sqrt((self._radius ** (-2))*(x**2 + y**2 ))

        #checking if the robot passes through the exact center of the obstacle,
        #because the gradient of the cost function is discontinuous at that point
        #if you are wondering why: plot the function on desmos
        if(d == 0):
            x += 0.1
            y += 0.1
            d = math.sqrt((self._radius ** (-2))*(x**2 + y**2 ))

        #checking if the obstacle and the circle the robot is bounded by intersect
        if(d < (self._radius + robotradius)):
            return self._penalty * (math.e) ** (-d)
        else:
            return 0
    def getCostGradient(self, robotstate, robotradius):
        x = self._x - robotstate[0]
        y = self._y - robotstate[1]

        d = math.sqrt((self._radius ** (-2))*(x**2 + y**2 ))

        if(d == 0):
            x += 0.1
            y += 0.1
            d = math.sqrt((self._radius ** (-2))*(x**2 + y**2 ))

        #checking if the obstacle and the circle the robot is bounded by intersect
        if(d < (self._radius + robotradius)):
            grad = (self._penalty/self._radius) * np.array([-(math.e ** (-d))*(x)/(math.sqrt(x**2 + y**2)), -(math.e ** (-d))*(y)/(math.sqrt(x**2 + y**2)), 0,0,0,0])
            return grad
        else:
            return np.zeros(6)

    def getCostHessian(self, robotstate, robotradius):
        '''
        This method represents the reason that one should use automatic differentiation
        '''
        x = self._x - robotstate[0]
        y = self._y - robotstate[1]

        d = math.sqrt((self._radius ** (-2))*(x**2 + y**2 ))

        if(d == 0):
            x += 0.1
            y += 0.1
            d = math.sqrt((self._radius ** (-2))*(x**2 + y**2 ))
        #checking if the obstacle and the circle the robot is bounded by intersect
        if(d < (self._radius + robotradius)):
            h = np.zeros((6,6))
            denominator = (((math.sqrt(x**2 + y**2)) ** 3) *(self._radius**2))
            h[0][0] = ((math.e**(-d))*(x**2)*(math.sqrt(x**2 + y**2)) - self._radius*(math.e**(-d))*(y**2))/denominator
            h[0][1] = x*((math.e**(-d))*y*(math.sqrt(x**2 + y**2)) + self._radius*(math.e**(-d))*y)/denominator
            h[1][0] = x*((math.e**(-d))*y*(math.sqrt(x**2 + y**2)) + self._radius*(math.e**(-d))*y)/denominator
            h[1][1] = ((math.e**(-d))*(y**2)*(math.sqrt(x**2 + y**2)) - self._radius*(math.e**(-d))*(x**2))/denominator

            h = self._penalty * h
            return h
        else:
            return np.zeros((6,6))

