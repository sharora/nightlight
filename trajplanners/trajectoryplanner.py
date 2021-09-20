class TrajectoryPlanner(object):
    def __init__(self, dynamics, costfunction, h, dt):
        """Abstract class for a trajectory optimizer

        Args:
           dynamics : class describing how state evolves over time
           costfunction : class describing cost at every state/action
           h : the number of steps (time horizon) to plan over
           dt : the delta time associated with every timestep
        """

        self._dynamics = dynamics
        self._costfunction = costfunction
        self._h = h
        self._dt = dt

    def optimizeTrajectory(self, u_init, x0):
        """Optimizes the trajectory given by x0, u_init to minimize cost over time

        Args:
           u_init : an array of initial controls
           x0 : the initial state in which the controls are applied from

        Returns:
           An array of controls that (locally) minimizes the cost incurred from x0

        """
        raise NotImplementedError()

