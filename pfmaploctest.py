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
from utils.utils import loadOCfromPath

import time

''' Test where Mecanum Robot Localizes to a Prior Map (occupancy grid) using a
Lidar Sensor to repeatedly update a particle filter. All units are in inches
unless otherwise specified. '''

'''-----CONFIGURATION STARTS-----'''

#map config
map_path = './maps/simple_grid.png'
map_cell_dim = 4

#lidar config
lidar_num_lasers = 81
lidar_range = 40
lidar_sample_frequency = 10 #in hertz
lidar_period = 1.0/lidar_sample_frequency

#robot config
x0 = np.array([72, 36, 0, 0, 0, 0])
robotwidth = 18

#particle filter config
numparticles = 200
particleGenMethod = ''
k_neff = 0.1

#visualizer settings
maxparticlesize = 15
minparticlesize = 1
viz_pixperinch = 6

#simulation config
dt = 0.02

'''-----CONFIGURATION ENDS -----'''

#creating map
base_map = loadOCfromPath(map_path)
oc = OccupancyGrid(base_map, map_cell_dim)

#creating objects for simulation
dynamics = MecanumDrive()
lidar = LidarSensor(lidar_num_lasers, lidar_range)

#creating particles and particle filter
states, weights = randomParticlesBox(numparticles, np.zeros(dynamics.xdim,),
                                  np.array([oc._width*oc._celldim,
                                            oc._width*oc._celldim, 0, 0, 0, 0]))
pf = ParticleFilter(states, weights, dynamics, lidar, k_neff)
viz = Visualizer(oc._width*map_cell_dim, oc._length*map_cell_dim, viz_pixperinch)


#lidar frequency tracking variables
lidar_wait = 0
graph_ls = False
#simulation
while True:
    #getting robot state from keyboard
    keys = viz.getKeyPresses()
    u = np.zeros((dynamics.udim,))
    u[0] = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 100
    u[1] = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 100

    #robot update step
    x0 = dynamics.step(x0, u, dt)
    pf.step(u, dt)

    #getting lidar measurement and updating particle filter
    if lidar_wait >= lidar_period:
        ls = lidar.getMeasurement(x0, oc)
        pf.updateweights(ls, oc)
        lidar_wait -= lidar_period
        graph_ls = True
    lidar_wait += dt

    #displaying current state in visualizer
    viz.graphOC(oc)

    #only graphing lidar when it is updated
    if graph_ls:
        viz.graphLidarScan(x0[0], x0[1], ls, lidar_range)
        graph_ls = False

    #getting states and weights
    # states = np.array(pf._states)
    states = pf._states
    weights = pf._weights

    xg = np.array(x0)

    #graphing the remaining items
    viz.graphSquareRobot(xg[0], xg[1], xg[2], robotwidth)
    viz.graphParticles(states, weights, maxparticlesize, minparticlesize)
    viz.updateFrame()

