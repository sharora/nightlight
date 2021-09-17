import numpy as np

class OccupancyGrid(object):
    def __init__(self, occupancygrid, celldim):
        super().__init__()
        self._oc = occupancygrid
        self._length = occupancygrid.shape[0]
        self._width = occupancygrid.shape[1]
        self._celldim = celldim #in inches

    def inBounds(self, x, y):
        """
        Args:
           x : x coordinate in inches
           y : y coordinate in inches

        Returns:
           true if the given (x,y) coordinate is in bounds of the region.
        """
        x = x / self._celldim
        y = y / self._celldim
        return x >= 0 and y >= 0 and x < self._width and y < self._length

    def isFree(self, x, y):
        """
        Args:
           x : x coordinate in inches
           y : y coordinate in inches

        Returns:
            True if the given coordinates (x,y) are empty, false otherwise
        """
        x = int(x / self._celldim)
        y = int(y / self._celldim)
        return not self._oc[x][y]

    def updateWithLidar(self, x, y, scan, maxrange):
        """Updates the map according to the given lidar scan.

        Args:
           x : the x coordinate in inches to update from
           y : the y coordinate in inches to update from
           scan : a lidar scan array
           maxrange : max range of lidar
        """
        #TODO handle different lidar and map cell sizes

        #preprocessing scan so that 1 represents occupied, 0 represents unoccupied
        scan = scan<0

        #bounds for occupancy grid
        lb = max(int(x) - maxrange, 0)
        rb = min(int(x) + maxrange + 1, self._width)
        db = max(int(y) - maxrange, 0)
        ub = min(int(y) + maxrange + 1, self._length)

        #bounds for scan
        lpad = max(0, maxrange-int(x))
        rpad = 2*maxrange + 1 - max(0, int(x) + maxrange + 1 - self._width)
        dpad = max(0, maxrange-int(y))
        upad = 2*maxrange + 1 - max(0, int(y) + maxrange + 1 - self._length)

        #adding scan to image
        self._oc[lb:rb, db:ub] += scan[lpad:rpad, dpad:upad]

        #clipping TODO remove this
        self._oc = np.clip(self._oc, 0, 1)



