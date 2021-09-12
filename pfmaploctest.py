from PIL import Image
import numpy as np
import pygame

from particlefilter.particlefilter import ParticleFilter
from particlefilter.particlefilter import Particle
from dynamics.mecanumdrive import MecanumDrive
from sensors.lidarsensor import LidarSensor
from visualizer.visualizer import Visualizer
from sandbox.occupancygrid import OccupancyGrid

'''
Test where Mecanum Robot Localizes to a Prior Map (occupancy grid) using
a Lidar Sensor to repeatedly update a particle filter. All units are in inches unless otherwise specified.
'''

'''-----CONFIGURATION STARTS-----'''
map_path = './maps/simple_grid.png'
map_cell_dim = 4

#TODO add this in
lidar_num_lasers = 41
lidar_sample_frequency = 10 #in hertz

numParticles = 200
particleGenMethod = ''


x0 = np.array([72, 72, 0, 0, 0, 0])

'''-----CONFIGURATION ENDS -----'''

#creating objects for simulation
dynamics = MecanumDrive()
lidar = LidarSensor(lidar_num_lasers)
# pf = ParticleFilter(particleList, dynamics, lidar, 0.1)

base_map = 1 - np.array(Image.open(map_path))
oc = OccupancyGrid(base_map, map_cell_dim)
viz = Visualizer(oc._width*map_cell_dim, oc._length*map_cell_dim, 6)

#simulation
while True:
    #setting controls using keyboard for testing
    keys = viz.getKeyPresses()
    u = np.zeros((3,))
    u[0] = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 50
    u[1] = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 50
    x0 = dynamics.step(x0, u, 0.02)

    ls = lidar.getMeasurement(x0, oc)
    viz.graphOC(oc)
    viz.graphCircularObstacle(x0[0], x0[1], 9)
    viz.graphLidarScan(x0[0], x0[1], ls)
    viz.updateFrame()

