from . costfunction import CostFunction
from jax import jit, grad, hessian
import jax.numpy as np
from functools import partial
import time

class MecanumObstacleCost(CostFunction):
    """
    Class Describing the cost function of a mecanum wheeled robot going to
    a target position in a region populated by obstacles.
    """
    def __init__(self, robot_radius, obstacles, Q, R, Qf):
        """
        Args:
           obstacles : an array of obstacle positions and radii [[x,y,r], ...]
           robot_radius : a radius that bounds the robot (whatever shape it is)
           Q : the cost matrix penalizing difference in state
           R : the cost matrix penalizing the control effort
           Q : the cost matrix penalizing difference in state at terminal state
        """
        super().__init__()
        self._robot_radius = robot_radius
        self._obstacles = obstacles
        self._Q = Q
        self._R = R
        self._Qf = Qf

        #defining gradient and hessian functions
        self.g_x = jit(grad(self.getCost, argnums=0))
        self.g_u = jit(grad(self.getCost, argnums=1))
        self.g_xx = jit(hessian(self.getCost, argnums=0))
        self.g_uu = jit(hessian(self.getCost, argnums=1))

        self.G_x = jit(grad(self.getTerminalCost, argnums=0))
        self.G_xx = jit(hessian(self.getTerminalCost, argnums=0))

    @partial(jit, static_argnums=(0,))
    def getCost(self, x, u, x_targ):
        """
        Args:
           x : the current state
           u : the action taken from the current state
           x_targ : the target state

        Returns:
           the cost incurred by the given state and action taken
        """
        totalcost = 0

        #distance cost
        err = x - x_targ
        totalcost += (np.transpose(err) @ self._Q @ err)

        #control cost
        totalcost += (np.transpose(u) @ self._R @ u)

        #obstacle cost for each obstacle
        for i in range(self._obstacles.shape[0]):
            totalcost += self.obstacleCost(x , self._obstacles[i])

        return totalcost

    @partial(jit, static_argnums=(0,))
    def obstacleCost(self, x, obstacle):
        """
        Args:
           x : the current stat
           obstacle : an obstacle represented by (x,y,r)

        Returns:
           the cost induced by the given obstacle on the robot at the
        current state
        """
        #finding the distance between the states
        dist = (x[0] - obstacle[0])**2 + (x[1] - obstacle[1])**2

        #checking if robot is inside obstacle
        radius = obstacle[2] + self._robot_radius
        indicator =  np.heaviside(radius**2 - dist, 0.5)

        c = dist / ((2*radius)**2)
        cost = (radius**4) * np.exp(-c) * indicator
        return cost

    @partial(jit, static_argnums=(0,))
    def getTerminalCost(self, x, x_targ):
        """
        Args:
           x : the terminal state

        Returns:
           the cost incurred at the terminal state

        """
        #distance cost
        err = x - x_targ
        return np.transpose(err) @ self._Qf @ err

if __name__ == '__main__':
    robot_radius = 9
    obstacles = np.array([[72., 72., 10.], [10., 10., 20.]])
    Q = np.array([
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]
        ])
    Qf = 10*np.array([
        [1., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0.1, 0., 0.],
        [0., 0., 0., 0., 0.1, 0.],
        [0., 0., 0., 0., 0., 0.1]
        ])
    R = 0.04*np.array([
        [1.0, 0., 0.],
        [0., 1.0, 0.],
        [0., 0., 1.0]
    ])

    x0 = np.array([80., 80., 0., 0., 0., 0.])
    x_targ = np.array([0., 0., 0., 0., 0., 0.])
    u = np.array([0., 0., 0.])

    c = MecanumObstacleCost(robot_radius, obstacles, Q, R, Qf)

    #gradient and hessian tests
    print(c.g_x(x0, u, x_targ))
    print(c.g_u(x0, u, x_targ))
    print(c.g_xx(x0, u, x_targ))
    print(c.g_uu(x0, u, x_targ))
    print(c.G_x(x0, x_targ))
    print(c.G_xx(x0, x_targ))

    start = time.time()
    print(c.getCost(x0, u, x_targ))
    print(time.time() - start)

    start = time.time()
    print(c.getCost(x0, u, x_targ))
    print(time.time() - start)

    start = time.time()
    print(c.getTerminalCost(x0, x_targ))
    print(time.time() - start)

    start = time.time()
    print(c.getTerminalCost(x0, x_targ))
    print(time.time() - start)
