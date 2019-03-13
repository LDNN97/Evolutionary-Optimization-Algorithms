import numpy as np
from .optimizer import *


class CMAESM(Optimizer):
    def __init__(self, func, count):
        super().__init__(func, count)
        self.nn = self.f.D

        self.xx = np.random.random(self.nn)
        self.xmean = np.copy(self.xx)
        self.sigma = 1

        self.lam = 4 + int(3 * np.log(self.nn))
        self.mu = int(self.lam / 2)
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        self.weights = np.array([w / np.sum(self.weights) for w in self.weights])
        self.mueff = 1 / np.sum(np.power(w, 2) for w in self.weights)

        self.cs = (self.mueff + 2) / (self.nn + self.mueff + 5)
        self.c1 = 2 / ((self.nn + 1.3) ** 2 + self.mueff)
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.nn + 2) ** 2 + self.mueff)])

        self.ps = np.zeros(self.nn)
        self.M = np.eye(self.nn)

    def step(self):
        # Sample
        pop = np.zeros((self.lam, self.nn))
        z = np.zeros((self.lam, self.nn))
        d = np.zeros((self.lam, self.nn))
        for i in range(self.lam):
            z[i] = np.random.normal(0, 1, self.nn)
            d[i] = np.dot(self.M, z[i])
            pop[i] = self.xmean + self.sigma * d[i]

        # sort and update mean
        fitvals = np.zeros(self.lam)
        for i in range(self.lam):
            fitvals[i] = self.f.evaluate(pop[i])
        argx = np.argsort(fitvals)
        if fitvals[argx[0]] < self.opti_f:
            self.opti_x = pop[argx[0]]
            self.opti_f = fitvals[argx[0]]
        self.xmean = self.xmean + self.sigma * np.sum(self.weights[i] * d[argx[i]] for i in range(self.mu))

        # update evolution path
        zz = np.sum(self.weights[i] * z[argx[i]] for i in range(self.mu))
        c = np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        self.ps -= self.cs * self.ps
        self.ps += c * zz

        # update matrix
        one = np.eye(self.nn, self.nn)
        part1 = one
        part2o = self.ps.reshape((self.nn, 1))
        part2t = self.ps.reshape((1, self.nn))
        part2 = self.c1 / 2 * (np.dot(part2o, part2t) - one)
        part3 = np.zeros((self.nn, self.nn))
        for i in range(self.mu):
            part3o = z[argx[i]].reshape((self.nn, 1))
            part3t = z[argx[i]].reshape((1, self.nn))
            part3 += self.weights[i] * np.dot(part3o, part3t)
        part3 = self.cmu / 2 * (part3 - one)
        self.M = np.dot(self.M, part1 + part2 + part3)

        # update step-size
        self.sigma *= np.exp((self.cs / 2) * (np.sum(np.power(x, 2) for x in self.ps) / self.nn - 1))

    def run(self):
        for i in range(self.maxgen):
            self.step()
            print(i, self.opti_f)

    def output(self):
        return self.opti_f, self.opti_x
