from PIL import Image
import numpy as np
import pygame

from particlefilter.particlefilter import ParticleFilter
from particlefilter.particlegenerator import randomParticlesBox
from particlefilter.particle import Particle
from dynamics.mecanumdrive import MecanumDrive
from sensors.lidarsensor import LidarSensor
from visualizer.visualizer import Visualizer
from sandbox.occupancygrid import OccupancyGrid

import time

'''
Test where Mecanum Robot Localizes to a Prior Map (occupancy grid) using
a Lidar Sensor to repeatedly update a particle filter. All units are in inches unless otherwise specified.
'''

'''-----CONFIGURATION STARTS-----'''
map_path = './maps/simple_grid.png'
map_cell_dim = 4

#TODO add this in
lidar_num_lasers = 81
lidar_range = 40
lidar_sample_frequency = 10 #in hertz

numparticles = 200
particleGenMethod = ''

dt = 0.02

x0 = np.array([72, 72, 0, 0, 0, 0])
robotwidth = 18

'''-----CONFIGURATION ENDS -----'''

#creating objects for simulation
dynamics = MecanumDrive()
lidar = LidarSensor(lidar_num_lasers, lidar_range)
particleList = randomParticlesBox(numparticles, np.zeros(6,), np.array([144, 144, 0, 0, 0, 0]))

pf = ParticleFilter(particleList, dynamics, lidar, 0.1)

base_map = 1 - np.array(Image.open(map_path))
oc = OccupancyGrid(base_map, map_cell_dim)
viz = Visualizer(oc._width*map_cell_dim, oc._length*map_cell_dim, 6)

#simulation
while True:
    #setting controls using keyboard for testing
    keys = viz.getKeyPresses()
    u = np.zeros((3,))
    u[0] = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 100
    u[1] = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 100
    x0 = dynamics.step(x0, u, dt)

    # pf.step(u, dt)

    ls = lidar.getMeasurement(x0, oc)

    # start = time.time()
    # pf.updateweights(ls, oc)
    # print(time.time() - start)
    # print(pf.getMaxParticle()._x)

    viz.graphOC(oc)
    viz.graphLidarScan(x0[0], x0[1], ls, lidar_range)
    viz.graphSquareRobot(x0[0], x0[1], x0[2], robotwidth)
    viz.updateFrame()

