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


