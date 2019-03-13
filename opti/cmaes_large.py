import numpy as np
from .optimizer import *


class CMAESL(Optimizer):
    def __init__(self, func, maxgen):
        super().__init__(func, maxgen)

        self.nn = self.f.D
        self.sigma = 0.2
        self.xmean = np.random.random(self.nn)

        self.lam = 4 + int(3 * np.log(self.nn))
        self.mm = 4 + int(3 * np.log(self.nn))
        self.mu = int(self.lam / 2)
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        self.weights = np.array([w / np.sum(self.weights) for w in self.weights])
        self.mueff = 1 / np.sum(np.power(w, 2) for w in self.weights)

        self.cs = 2 * self.lam / self.nn
        self.c_d = np.array([1/(np.power(1.5, i) * self.nn) for i in range(self.mm)])
        self.c_c = np.array([self.lam/(np.power(4, i) * self.nn) for i in range(self.mm)])

        self.gen = 0
        self.ps = np.zeros(self.nn)
        self.m = np.zeros((self.mm, self.nn))

    def step(self):
        # sample
        pop = np.zeros((self.mm, self.nn))
        z = np.zeros((self.mm, self.nn))
        d = np.zeros((self.mm, self.nn))

        fitvals = np.zeros(self.mm)
        for i in range(self.mm):
            z[i] = np.random.normal(0, 1, self.nn)
            d[i] = z[i]
            for j in range(min(self.gen, self.mm)):
                d[i] = (1 - self.c_d[j]) * d[i] + \
                    self.c_d[j] * self.m[j] * (np.dot(self.m[j].T, d[i]))
            pop[i] = self.xmean + self.sigma * d[i]
            # boundary check

        # select
        for i in range(self.mm):
            fitvals[i] = self.f.evaluate(pop[i])
        argx = np.argsort(fitvals)
        if fitvals[argx[0]] < self.opti_f:
            self.opti_x = pop[argx[0]]
            self.opti_f = fitvals[argx[0]]

        # update
        # mean
        self.xmean = self.xmean + self.sigma * np.sum(self.weights[i] * d[argx[i]] for i in range(self.mu))

        # evolution path
        zz = np.sum(self.weights[i] * z[argx[i]] for i in range(self.mu))
        c = np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        self.ps = (1 - self.cs) * self.ps + c * zz

        # covariance matrix
        for i in range(self.mm):
            c = np.sqrt(self.mueff * self.c_c[i] * (2 - self.c_c[i]))
            self.m[i] = (1 - self.c_c[i]) * self.m[i] + c * zz

        # step-size
        self.sigma *= np.exp((self.cs / 2) * (np.sum(np.power(x, 2) for x in self.ps) / self.nn - 1))

        self.gen += 1

    def run(self):
        for i in range(self.maxgen):
            self.step()
            print(i, self.opti_f)

    def output(self):
        return self.opti_f, self.opti_x
