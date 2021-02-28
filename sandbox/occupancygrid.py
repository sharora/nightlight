import numpy as np

class OccupancyGrid(object):
    def __init__(self, occupancygrid, celldim):
        super().__init__()
        self._oc = occupancygrid
        self._length = occupancygrid.shape[0]
        self._width = occupancygrid.shape[1]
        self._celldim = celldim
