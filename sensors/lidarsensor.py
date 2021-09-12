import numpy as np
import math
from . sensor import Sensor

class LidarSensor(Sensor):
    def __init__(self, numofLasers):
        Sensor.__init__(self)
        assert numofLasers % 2 == 1, "odd number of lasers needed"
        self._numofLasers = numofLasers

        #assuming lasers are equally spaced
        self._thetalist = []
        laserspacing = 2*math.pi/numofLasers
        for i in range(numofLasers):
            self._thetalist.append(i*laserspacing)

    def getMeasurement(self, x, oc):
        laserscan = []

        #loop through all lines starting at coordinate
        for theta in self._thetalist:
            currx = x[0]
            curry = x[1]
            laserscan.append([])

            #finding each increment
            xinc = math.cos(theta)
            yinc = math.sin(theta)

            #normalizing increment so one of them is 1
            larger = max(abs(xinc), abs(yinc))
            xinc /= larger
            yinc /= larger

            while oc.inBounds(currx, curry) and oc.isFree(currx, curry):
                laserscan[-1].append(np.array([currx - x[0], curry - x[1]]))
                currx += xinc
                curry += yinc
            laserscan[-1].append(np.array([currx - x[0], curry - x[1]]))

        return laserscan

    def getMeasurementProbability(self, xt, ls, oc):
        '''
        In this case we return a similarity score instead of a probability.
        '''
        score = 0
        #loop over all lasers
        for laser in ls:
            for cell in laser:
                xls = cell[0] + xt[0]
                yls = cell[1] + xt[0]
                if not oc.inBounds(xls, yls):
                    continue
                if laser[-1] == cell and not oc.isFree(xls, yls):
                    score += 1
                elif oc.isFree(xls, yls):
                    score += 1
        return math.exp(score)


