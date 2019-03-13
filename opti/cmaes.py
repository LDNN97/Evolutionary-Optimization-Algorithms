import numpy as np
from .optimizer import *


class CMAES(Optimizer):
    def __init__(self, func, count):
        super().__init__(func, count)

        self.nn = self.f.D
        self.xx = np.random.random(self.nn)
        self.xmean = np.copy(self.xx)
        self.sigma = 1

        self.lam = 4 + int(3 * np.log(self.nn))
        self.mu = int(self.lam / 2)
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        self.weights = np.array([w / sum(self.weights) for w in self.weights])
        self.mueff = 1 / np.sum(np.power(w, 2) for w in self.weights)

        self.cc = (4 + self.mueff / self.nn) / (self.nn + 4 + 2 * self.mueff / self.nn)
        self.cs = (self.mueff + 2) / (self.nn + self.mueff + 5)
        self.c1 = 2 / ((self.nn + 1.3) ** 2 + self.mueff)
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.nn + 2) ** 2 + self.mueff)])
        self.damps = 1 + self.cs + 2 * max([0, ((self.mueff - 1) / self.nn) ** 0.5 - 1])

        self.pc, self.ps = np.zeros(self.nn), np.zeros(self.nn)
        self.B = np.eye(self.nn)
        self.D = np.ones(self.nn)
        self.C = np.eye(self.nn)
        self.M = np.eye(self.nn)

    def step(self):
        # Sample
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = self.D ** 0.5
        self.M = self.B * self.D
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
        dd = np.sum(self.weights[i] * d[argx[i]] for i in range(self.mu))
        c = np.sqrt(self.cc * (2 - self.cc) * self.mueff)
        self.pc -= self.cc * self.pc
        self.pc += c * dd

        # update covariance matrix
        part1 = (1 - self.c1 - self.cmu) * self.C
        part2o = self.pc.reshape(self.nn, 1)
        part2t = self.pc.reshape(1, self.nn)
        part2 = self.c1 * np.dot(part2o, part2t)
        part3 = np.zeros((self.nn, self.nn))
        for i in range(self.mu):
            part3o = d[argx[i]].reshape(self.nn, 1)
            part3t = d[argx[i]].reshape(1, self.nn)
            part3 += self.cmu * self.weights[i] * np.dot(part3o, part3t)
        self.C = part1 + part2 + part3

        # update step-size
        self.sigma *= np.exp((self.cs / 2) * (np.sum(np.power(x, 2) for x in self.ps) / self.nn - 1))

    def run(self):
        for i in range(self.maxgen):
            self.step()
            print(i, self.opti_f)

    def output(self):
        return self.opti_f, self.opti_x
