import numpy as np
from .optimizer import *


class DE(Optimizer):
    def __init__(self, func, count, popsize=50):
        super().__init__(func, count)
        self.popsize = popsize
        self.pop = np.random.random((self.popsize, self.f.D))
        self.fit = np.zeros(self.popsize)
        for i in range(self.popsize):
            self.fit[i] = self.f.evaluate(self.pop[i])

    def step(self):
        newpop = np.zeros((self.popsize, self.f.D))
        newfit = np.zeros(self.popsize)

        cr, f = np.random.random(), np.random.random()
        for i in range(self.popsize):
            ind = np.random.choice(np.arange(self.popsize), 3, replace=False)
            j = np.random.randint(self.f.D)

            trial = np.zeros(self.f.D)
            for k in range(self.f.D):
                if (np.random.random() < cr) or (k == self.f.D):
                    trial[j] = self.pop[ind[0]][j] + f * (self.pop[ind[1]][j] - self.pop[ind[2]][j])
                    # if trial[j] > 1:
                    #     trial[j] = self.pop[i][j] + np.random.random() * (1 - self.pop[i][j])
                    # if trial[j] < 0:
                    #     trial[j] = np.random.random() * self.pop[i][j]
                else:
                    trial[j] = self.pop[i][j]
                j = (j + 1) % self.f.D

            trial_value = self.f.evaluate(trial)
            newpop[i] = trial if trial_value <= self.fit[i] else self.pop[i]
            newfit[i] = trial_value if trial_value <= self.fit[i] else self.fit[i]

            [self.opti_f, self.opti_x] = [newfit[i], newpop[i]] if newfit[i] <= self.opti_f \
                else [self.opti_f, self.opti_x]

        self.pop = newpop
        self.fit = newfit

    def run(self):
        for i in range(self.maxgen):
            self.step()
            print(i, self.opti_f)

    def output(self):
        return self.opti_f, self.opti_x
