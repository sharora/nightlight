import numpy as np

class CircularObstacle(object):
    def __init__(self, x, y, radius, penalty):
        super().__init__()
        self._x = x
        self._y = y
        self._radius = radius
        self._penalty = penalty

    def getCost(self, robotstate):
        #TODO add cost function that is inversely proportional to distance
        # clips to zero, when robot is outside radius
        pass
