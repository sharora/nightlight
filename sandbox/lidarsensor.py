import numpy as np
import math
from sensor import Sensor

class LidarSensor(Sensor):
    def __init__(self, numofLasers):
        Sensor.__init__(self)
        self._numofLasers = numofLasers

        #assuming lasers are equally spaced
        self._thetalist = []
        laserspacing = 2*math.pi/numofLasers
        for i in range(numofLasers):
            self._thetalist.append(i*laserspacing)

    def getMeasurement(self, x, oc):
        #oc is the occupancy grid
        #x is the current robot state
        laserscan = []
        #round robot coordinate to map pos
        mapx = int(x[0]/oc._celldim)
        mapy = int(x[1]/oc._celldim)

        #setting robot coordinate to empty
        # laserscan[oc._length-mapy-1][mapx] = 1
        increment = 0

        #loop through all lines starting at coordinate
        for i in range(len(self._thetalist)):
            #handle pi/2 and 3pi/2
            m = math.tan(self._thetalist[i])
            currx = mapx + 0.5
            curry = mapy + 0.5
            laserscan.append([])
            if(m > 1 or m < -1):
                if(math.sin(self._thetalist[i]) > 0):
                    increment = 1
                else:
                    increment = -1
                while(oc._oc[oc._length - int(curry) - 1][int(currx)] != 0 and currx > 0 and curry > 0 and currx < oc._width and curry < oc._length):
                    # laserscan[oc._length - int(curry) - 1][int(currx)] = 1
                    laserscan[i].append([int(currx) - mapx, int(curry) - mapy])
                    currx += 1/m*increment
                    curry += 1*increment
                laserscan[i].append([int(currx) - mapx, int(curry) - mapy])
                # laserscan[oc._length - int(curry) - 1][int(currx)] = 0
            else:
                if(math.cos(self._thetalist[i]) > 0):
                    increment = 1
                else:
                    increment = -1
                while(oc._oc[oc._length - int(curry) - 1][int(currx)] != 0 and currx > 0 and curry > 0 and currx < oc._width and curry < oc._length):
                    # laserscan[oc._length - int(curry) - 1][int(currx)] = 1
                    laserscan[i].append([int(currx) - mapx, int(curry) - mapy])
                    currx += 1*increment
                    curry += m*increment
                laserscan[i].append([int(currx) - mapx, int(curry) - mapy])
                # laserscan[oc._length - int(curry) - 1][int(currx)] = 0
        # laserscan[oc._lengtmapy-1][mapx] = 2

        return laserscan
    def getMeasurementProbability(self, xt, ls, oc):
        score = 0
        for i in range(len(ls)):
            for j in range(len(ls[i])):
                xls = ls[i][j][0] + int(xt[0]/4)
                yls = ls[i][j][1] + int(xt[1]/4)
                if(xls < 0 or yls < 0 or yls > oc._length - 1 or xls > oc._width - 1):
                    continue
                if(j == len(ls[i]) - 1):
                    if(oc._oc[oc._length - yls - 1][xls] == 0):
                        score += 1
                else:
                    if(oc._oc[oc._length - yls - 1][xls] == 1):
                        score += 1
        score = math.exp(0.5*score)
        return score


