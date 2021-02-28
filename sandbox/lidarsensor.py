import numpy as np
import math

class LidarSensor(object):
    def __init__(self, numofLasers):
        super().__init__()
        self._numofLasers = numofLasers

        #assuming lasers are equally spaced
        self._thetalist = []
        laserspacing = 2*math.pi/numofLasers
        for i in range(numofLasers):
            self._thetalist.append(i*laserspacing)

    def getMeasurement(self, oc, x):
        #copy map and fill map with -1s
        laserscan = np.copy(oc._oc)
        for i in range(oc._length):
            for j in range(oc._width):
                laserscan[i][j] = -1
        #round robot coordinate to map pos
        mapx = int(x[0]/oc._celldim)
        mapy = int(x[1]/oc._celldim)

        #setting robot coordinate to empty
        laserscan[oc._length-mapy-1][mapx] = 1
        increment = 0

        #loop through all lines starting at coordinate
        for i in range(len(self._thetalist)):
            m = math.tan(self._thetalist[i])
            currx = mapx + 0.5
            curry = mapy + 0.5
            if(m > 1 or m < -1):
                if(math.sin(self._thetalist[i]) > 0):
                    increment = 1
                else:
                    increment = -1
                while(oc._oc[oc._length - int(curry) - 1][int(currx)] != 0 and currx > 0 and curry > 0 and currx < oc._width and curry < oc._length):
                    laserscan[oc._length - int(curry) - 1][int(currx)] = 1
                    currx += 1/m*increment
                    curry += 1*increment
                laserscan[oc._length - int(curry) - 1][int(currx)] = 0
            else:
                if(math.cos(self._thetalist[i]) > 0):
                    increment = 1
                else:
                    increment = -1
                while(oc._oc[oc._length - int(curry) - 1][int(currx)] != 0 and currx > 0 and curry > 0 and currx < oc._width and curry < oc._length):
                    laserscan[oc._length - int(curry) - 1][int(currx)] = 1
                    currx += 1*increment
                    curry += m*increment
                laserscan[oc._length - int(curry) - 1][int(currx)] = 0

        return laserscan
    def getLaserScanCorrelation(self, l1, l2):
        score = 0
        for i in range(l1.shape[0]):
            for j in range(l1.shape[1]):
                if(l1[i][j] == 1 and l2[i][j] == 1):
                    score += 1
                if(l1[i][j] == 0 and l2[i][j] == 0):
                    score += 1
        return score


