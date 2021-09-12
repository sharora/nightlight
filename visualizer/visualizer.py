import pygame
import cv2
import numpy as np

class Visualizer(object):
    """
    Class to Visualize a 2d Environment containing mobile robots, obstacles,
    occupancy grids, and lidar scans.
    """
    def __init__(self, width, height, pixperinch):
        """
        Args:
           width : width of region in inches
           length : length of region in inches
           pixperinch : number of pixels to use per inch
        """
        self._width = width
        self._height = height
        self._pixperinch = pixperinch

        pygame.init()
        self._display = pygame.display.set_mode((width*pixperinch,
                                                 height*pixperinch))

    def graphOC(self, oc):
        '''
        Graphs the given occupancy grid oc
        '''
        #making the frame and scaling it up
        im = np.stack([0*oc._oc, 140*oc._oc, 255*oc._oc], -1)
        im = np.flip(im, 1)
        scaling = oc._celldim * self._pixperinch
        im = cv2.resize(im, (0,0), fx=scaling, fy=scaling,
                        interpolation=cv2.INTER_NEAREST)

        #displaying the occupancy grid
        surf = pygame.surfarray.make_surface(im)
        self._display.blit(surf, (0,0))

    def graphSquareRobot(self, x, y, theta, l):
        """Graphs a square robot with pose (x,y,theta) and sidelength l

        Args:
           x : x coordinate in inches
           y : y coordinate in inches
           theta : robot angle in radians
           l : sidelength in inches
        """
        pass

    def graphCircularObstacle(self, x, y, radius):
        """
        Args:
           x : x coordinate in inches
           y : y coordinate in inches
           radius : radius in inches
        """
        pygame.draw.circle(self._display, (255, 0, 0),
                           self.imageCoordinates(x,y), radius*self._pixperinch)

    def graphLidarScan(self, x, y, scan):
        """Graphs a lidar scan originating at (x,y)

        Args:
           self arg1
           x : x coordinate of scan in inches
           y : y coordinate of scan in inches
           scan : list of distances in each direction
        """
        im = np.zeros((self._width, self._height, 3))

        #loop over all lasers
        for laser in scan:
            for cell in laser:
                cellx = int(cell[0] + x)
                celly = int(cell[1] + y)
                im[cellx, celly] = (0, 255, 0)

        #scaling up the scan
        im = cv2.resize(im, (0,0), fx=self._pixperinch, fy=self._pixperinch,
                        interpolation=cv2.INTER_NEAREST)
        im = np.flip(im, 1)

        #displaying the occupancy grid
        surf = pygame.surfarray.make_surface(im)
        surf.set_colorkey((0,0,0))
        self._display.blit(surf, (0,0))


    def updateFrame(self):
        """
        Updates the display to reflect graphing changes
        """
        pygame.display.update()

    def imageCoordinates(self, x, y):
        """Returns the coordinates in the image frame from the grid coordinates

        Args:
           self arg1
           x : x coordinate in inches
           y : y coordinate in inches

        Returns:
           Tuple (x', y') of transformed coordinates
        """
        return (self._pixperinch*x, self._pixperinch*(self._height - y))

    def getKeyPresses(self):
        '''
        Returns a dictionary of boolean values, one for each key, indicating
        whether the key is pressed or not. Useful for debugging.
        '''
        pygame.event.pump()
        return pygame.key.get_pressed()

if __name__ == '__main__':
    from PIL import Image
    import sys

    sys.path.append('../')
    from sandbox.occupancygrid import OccupancyGrid

    base_map = 1 - np.array(Image.open('../maps/simple_grid.png'))
    oc_cell_dim = 4
    oc = OccupancyGrid(image, oc_cell_dim)
    viz = Visualizer(oc._width*oc_cell_dim, oc._length*oc_cell_dim, 6)

    while True:
        viz.graphOC(oc)
        viz.graphCircularObstacle(72, 72 , 9)
        viz.updateFrame()


