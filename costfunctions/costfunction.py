class CostFunction(object):
    """Abstract class describing a cost function and providing useful tools
    like automatic differentiating to find the gradient and hessian.
    """
    def __init__(self):
        super().__init__()

    def getCost(self, x, u, x_targ):
        """
        Args:
           x : the current state
           u : the action taken from the current state
           x_targ : the target state

        Returns:
           the cost incurred by the given state and action taken
        """
        raise NotImplementedError()

    def getTerminalCost(self, x):
        """
        Args:
           x : the terminal state

        Returns:
           the cost incurred at the terminal state

        """
        raise NotImplementedError()


