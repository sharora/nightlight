from . trajectoryplanner import TrajectoryPlanner
from jax import jit, grad
import jax.numpy as np
from functools import partial

class GradientDescentPlanner(TrajectoryPlanner):
    """
    Optimizes Trajectories using Gradient Descent
    """
    def __init__(self, dynamics, costfunction, h, dt, lr):
        """
        Args:
           dynamics : class describing how state evolves over time
           costfunction : class describing cost at every state/action
           h : time horizon to plan over
           dt : the delta time associated with every timestep
        """
        super().__init__(dynamics, costfunction, h, dt)

        self._lr = lr

        #creating jit gradient function
        self._u_grad = jit(grad(self.simulateRollout, argnums=1))

    def optimizeTrajectory(self, u_init, x0, x_targ, gdsteps):
        """Optimizes the trajectory given by x0, u_init to minimize cost over time

        Args:
           u_init : an array of initial controls
           x0 : the initial state in which the controls are applied from
           x_targ : the target state
           gdsteps : the number of gradient steps to take

        Returns:
           An array of controls that (locally) minimizes the cost incurred from x0

        """
        #main optimization loop
        for i in range(gdsteps):
            #calculating the gradient
            u_grad = self._u_grad(x0, u_init, x_targ)

            #updating the weights
            u_init -= self._lr*u_grad

        return u_init

    @partial(jit, static_argnums=(0,))
    def simulateRollout(self, x0, u, x_targ):
        """Simulates a trajectory rollout

        Args:
           x0 : the initial state
           u : an array of controls
           x_targ : the target state

        Returns:
            Returns the cost incurred along the Trajectory

        """
        #TODO can jax jit this, probably not? think of solution
        totalcost = 0
        for i in range(self._h):
            #adding to cost
            totalcost += self._costfunction.getCost(x0, u[i], x_targ)

            #forward dynamics (applying control)
            x0 = self._dynamics.step(x0, u[i], self._dt)

        #adding terminal cost
        totalcost += self._costfunction.getTerminalCost(x0, x_targ)

        return totalcost


