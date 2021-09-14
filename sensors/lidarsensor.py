import numpy as np
from . sensor import Sensor

class LidarSensor(Sensor):
    def __init__(self, numofLasers, maxrange):
        Sensor.__init__(self)
        assert numofLasers % 2 == 1, "odd number of lasers needed"
        self._numofLasers = numofLasers
        self._range = maxrange

        #assuming lasers are equally spaced
        self._thetalist = np.linspace(0, 2*np.pi, num=numofLasers, endpoint=False)

    def getMeasurement(self, x, oc):
        laserscan = np.zeros((2*self._range + 1, 2*self._range + 1))

        xround = int(x[0])
        yround = int(x[1])
        #loop through all lines starting at coordinate
        for theta in self._thetalist:
            currx = int(x[0])
            curry = int(x[1])

            #finding each increment
            xinc = np.cos(theta)
            yinc = np.sin(theta)

            #normalizing increment so one of them is 1
            larger = max(abs(xinc), abs(yinc))
            xinc /= larger
            yinc /= larger

            #tracking distance of beam
            distinc = np.sqrt(xinc**2 + yinc**2)
            currdist = 0

            #main section of lidar scan
            while oc.inBounds(currx, curry) and oc.isFree(currx, curry) and currdist < self._range:
                laserscan[int(currx - xround + self._range), int(curry - yround + self._range)] = 1
                currx += xinc
                curry += yinc
                currdist += distinc

            #writing the final value of the scan
            final = -1
            if oc.inBounds(currx, curry) and oc.isFree(currx, curry):
                final = 1
            laserscan[int(currx - xround + self._range), int(curry - yround + self._range)] = final

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
                if np.all(laser[-1] == cell) and not oc.isFree(xls, yls):
                    score += 1
                elif oc.isFree(xls, yls):
                    score += 1
        return score


