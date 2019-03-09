class Optimizer(object):
    def __init__(self, func, maxgen, *arg1, **arg2):
        """
        initialize attributes of the instance
        :argument
            self.f : problems to be solved
            self.count : max generation number
            self.opti_x : optimal solution
            self.opti_f : optimal value
        """
        self.f = func
        self.maxgen = maxgen
        self.opti_x = []
        self.opti_f = 1e10

    def step(self):
        """
        a single step of the evolution process
        """
        pass

    def run(self):
        """
        control the generation number
        """
        pass

    def output(self):
        """
        return the solution and optimal value
        return:
            self.opti_x, self.opti_f
        """
        pass
