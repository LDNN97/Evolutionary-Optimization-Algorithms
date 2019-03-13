class Problem(object):
    def __init__(self, dim, lb, up, *arg1, **arg2):
        """
        initialize attributes of the problem
        :argument
            D : Dimension
            lb : lower bound
            ub : upper bound
        """
        self.D = dim
        self.lb = lb
        self.ub = up

    def evaluate(self, x):
        """
        define the evaluate methods
        :argument
            x : input
        """
        pass

