from . trajectoryplanner import TrajectoryPlanner
from jax import jit, grad
import jax.numpy as np
from functools import partial
import time

class iLQRPlanner(TrajectoryPlanner):
    """
    Optimizes Trajectories using iterative Linear Quadratic Regulator
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

    def optimizeTrajectory(self, u_init, x0, x_targ, steps):
        """Optimizes the trajectory given by x0, u_init to minimize cost over time

        Args:
           u_init : an array of initial controls
           x0 : the initial state in which the controls are applied from
           x_targ : the target state
           gdsteps : the number of gradient steps to take

        Returns:
           An array of controls that (locally) minimizes the cost incurred from x0

        """
        #defining initial controls
        xlis = np.zeros((self._h, self._dynamics.xdim))
        ulis = u_init
        K = np.zeros((self._h, self._dynamics.udim, self._dynamics.xdim))
        d = np.zeros((self._h, self._dynamics.udim))

        #main optimization loop
        for i in range(100):
            #forward rollout
            cost, xlis, ulis, f_x, f_u, g_x, g_u, g_xx, g_uu, p, P = self.simulateRollout(x0, x_targ, xlis, ulis, K, d)
            print(cost)

            #update controls
            K,d = self.updateControls(f_x, f_u, g_x, g_u, g_xx, g_uu, p, P)

        return ulis

    # @partial(jit, static_argnums=(0))
    def updateControls(self, f_x, f_u, g_x, g_u, g_xx, g_uu, p, P):
        """
        Args:
        """
        Klis = []
        dlis = []
        #starting at final control and working backwards
        for i in range(self._h):
            t = self._h - i - 1

            #calculating action-value function gradients/hessians
            q_x = g_x[t] + f_x[t].T @ p
            q_u = g_u[t] + f_u[t].T @ p

            #TODO include cross g terms in equation (assuming they don't exist for now)
            q_xx = g_xx[t] + f_x[t].T @ P @ f_x[t]
            q_xu = f_x[t].T @ P @ f_u[t]
            q_ux = f_u[t].T @ P @ f_x[t]
            q_uu = g_uu[t] + f_u[t].T @ P @ f_u[t]

            #calculating optimal control deltas (K and d terms)
            K = np.linalg.inv(q_uu) @ q_ux
            d = np.linalg.inv(q_uu) @ q_u
            Klis.insert(0, K)
            dlis.insert(0, d)

            #calculating the new value function expansion (for the current step)
            P = q_xx + K.T @ q_uu @ K - q_xu @ K - K.T @ q_ux
            p = q_x - K.T @ q_u + K.T @ q_uu @ d - q_xu @ d

        Klis = np.array(Klis)
        dlis = np.array(dlis)
        return Klis, dlis

    @partial(jit, static_argnums=(0,))
    def simulateRollout(self, x0, x_targ, xlis, ulis, K, d):
        """Simulates a trajectory rollout

        Args:
           x0 : the initial state
           u : an array of controls
           x_targ : the target state

        Returns:
            Returns the cost incurred along the Trajectory

        """
        new_xlis = []
        new_ulis = []

        f_x = []
        f_u = []

        g_x = []
        g_u = []
        g_xx = []
        g_uu = []

        totalcost = 0
        for i in range(self._h):
            u = ulis[i] - 0.01*d[i] - K[i] @ (xlis[i] - x0)

            #adding to cost
            totalcost += self._costfunction.getCost(x0, u, x_targ)

            #tracking all derivatives
            f_x.append(self._dynamics.getA(x0, u))
            f_u.append(self._dynamics.getB(x0, u))
            g_x.append(self._costfunction.g_x(x0, u, x_targ))
            g_u.append(self._costfunction.g_u(x0, u, x_targ))
            g_xx.append(self._costfunction.g_xx(x0, u, x_targ))
            g_uu.append(self._costfunction.g_uu(x0, u, x_targ))

            #tracking current state and controls
            new_xlis.append(x0)
            new_ulis.append(u)

            #forward dynamics (applying control)
            x0 = self._dynamics.step(x0, u, self._dt)

        #adding terminal cost
        totalcost += self._costfunction.getTerminalCost(x0, x_targ)

        #setting final value function expansions
        p = self._costfunction.G_x(x0, x_targ)
        P = self._costfunction.G_xx(x0, x_targ)

        #converting to jnp arrays to return
        xlis = np.array(new_xlis)
        ulis = np.array(new_ulis)

        f_x = np.array(f_x)
        f_u = np.array(f_u)

        g_x = np.array(g_x)
        g_u = np.array(g_u)
        g_xx = np.array(g_xx)
        g_uu = np.array(g_uu)

        return totalcost, xlis, ulis, f_x, f_u, g_x, g_u, g_xx, g_uu, p, P

