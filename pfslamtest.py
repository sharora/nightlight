from PIL import Image
import numpy as np
import pygame

from particlefilter.particlefilter import ParticleFilter
from particlefilter.particlegenerator import uniformParticles
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

#ground truth map config
map_path = './maps/simple_grid.png'
map_cell_dim = 4

#generated map config
new_map_dim = (200, 200)
new_map_cell_dim = 1

#lidar config
lidar_num_lasers = 81
lidar_range = 40
lidar_sample_frequency = 50 #in hertz
lidar_period = 1.0/lidar_sample_frequency

#robot config
x0 = np.array([72, 36, 0, 0, 0, 0]) #ground truth position
x0_newmap = np.array([100, 100, 0, 0, 0, 0])
robotwidth = 18

#particle filter config
numparticles = 200
particleGenMethod = ''
k_neff = 0.1

#visualizer settings
maxparticlesize = 50
minparticlesize = 1
viz_pixperinch = 6

#simulation config
dt = 0.02

'''-----CONFIGURATION ENDS -----'''

#creating ground truth map
base_map = loadOCfromPath(map_path)
oc = OccupancyGrid(base_map, map_cell_dim)

#creating the generated map
new_map = np.zeros(new_map_dim)
new_oc = OccupancyGrid(new_map, new_map_cell_dim)

#creating objects for simulation
dynamics = MecanumDrive()
lidar = LidarSensor(lidar_num_lasers, lidar_range)

#creating particles and particle filter
states, weights = uniformParticles(numparticles, x0_newmap)
pf = ParticleFilter(states, weights, dynamics, lidar, k_neff)
viz = Visualizer(new_oc._width*new_map_cell_dim, new_oc._length*new_map_cell_dim, viz_pixperinch)


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
    x0_newmap = dynamics.step(x0_newmap, u, dt)
    pf.step(u, dt)

    #getting lidar measurement and updating particle filter
    if lidar_wait >= lidar_period:
        ls = lidar.getMeasurement(x0, oc)
        pf.updateweights(ls, new_oc)
        lidar_wait -= lidar_period
        graph_ls = True
    lidar_wait += dt

    maxParticleState = pf.getMaxParticle()[0]
    if graph_ls:
        #updating map using the particle with the highest weight
        new_oc.updateWithLidar(maxParticleState[0], maxParticleState[1], ls, lidar_range)

    #displaying current state in visualizer
    viz.graphOC(new_oc)

    #only graphing lidar when it is updated
    if graph_ls:
        viz.graphLidarScan(maxParticleState[0], maxParticleState[1], ls, lidar_range)
        graph_ls = False

    #getting states and weights
    states = pf._states
    weights = pf._weights

    #graphing the remaining items
    viz.graphSquareRobot(x0_newmap[0], x0_newmap[1], x0_newmap[2], robotwidth)
    viz.graphParticles(states, weights, maxparticlesize, minparticlesize)
    viz.updateFrame()
