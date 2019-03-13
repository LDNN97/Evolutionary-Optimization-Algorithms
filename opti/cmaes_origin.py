import copy
import numpy as np
from .optimizer import *


class CMAESO(Optimizer):
    def __init__(self, func, count):
        super().__init__(func, count)

        self.nn = self.f.D
        self.xx = np.random.random(self.nn)
        self.xmean = self.xx[:]
        self.sigma = 0.2

        self.lam = 4 + int(3 * np.log(self.nn))
        self.mu = int(self.lam / 2)
        self.weights = [np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)]
        self.weights = [w / sum(self.weights) for w in self.weights]
        self.mueff = sum(self.weights) ** 2 / sum(w ** 2 for w in self.weights)  # ???

        self.cc = (4 + self.mueff / self.nn) / (self.nn + 4 + 2 * self.mueff / self.nn)
        self.cs = (self.mueff + 2) / (self.nn + self.mueff + 5)
        self.c1 = 2 / ((self.nn + 1.3) ** 2 + self.mueff)
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.nn + 2) ** 2 + self.mueff)])
        self.damps = 1 + self.cs + 2 * max([0, ((self.mueff - 1) / self.nn) ** 0.5 - 1])  # ???

        self.pc, self.ps = np.zeros(self.nn), np.zeros(self.nn)
        self.B = np.eye(self.nn)
        self.D = np.ones(self.nn)
        self.C = np.eye(self.nn)
        self.invsqrtc = np.eye(self.nn)

    def step(self):
        # Sample
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = self.D ** 0.5
        for i in range(self.nn):
            for j in range(self.nn):
                self.invsqrtc[i][j] = sum(self.B[i][k] * self.B[j][k] / self.D[k] for k in range(self.nn))

        newpop = []
        for i in range(self.lam):
            z = self.D * np.random.normal(0, 1, len(self.D))
            nn = self.xmean + self.sigma * np.dot(self.B, z)
            newpop.append(nn)

        # Selection and Recombination
        xmeanold = copy.deepcopy(self.xmean)
        fitvals = []
        for xx in newpop:
            fit = self.f.evaluate(xx)
            fitvals.append(fit)
        argx = np.argsort(fitvals)
        if fitvals[argx[0]] < self.opti_f:
            self.opti_x = newpop[argx[0]]
            self.opti_f = fitvals[argx[0]]
        self.xmean = sum(self.weights[j] * newpop[argx[j]] for j in range(self.mu))

        #        for i in range(self.nn):
        #            f.write(str(self.xmean[i]) + ' ')
        #        f.write('\n')

        # update evolution path
        y = self.xmean - xmeanold
        z = np.dot(self.invsqrtc, y)
        c = (self.cs * (2 - self.cs) * self.mueff) ** 0.5 / self.sigma
        self.ps -= self.cs * self.ps
        self.ps += c * z
        c = (self.cc * (2 - self.cc) * self.mueff) ** 0.5 / self.sigma
        self.pc -= self.cc * self.pc
        self.pc += c * y

        #        for i in range(self.nn):
        #            f.write(str(y[i]/self.sigma) + ' ')
        #        f.write('\n')
        #        for i in range(self.nn):
        #            f.write(str(self.ps[i]) + ' ')
        #        f.write('\n')
        #        for i in range(self.nn):
        #            f.write(str(self.pc[i]) + ' ')
        #        f.write('\n')

        # update covariance matrix
        c1a = self.c1
        for i in range(self.nn):
            for j in range(self.nn):
                cmuij = sum(self.weights[k] * (newpop[argx[k]][i] - xmeanold[i])
                            * (newpop[argx[k]][j] - xmeanold[j]) for k in range(self.mu)) / self.sigma ** 2
                self.C[i][j] += (-c1a - self.cmu) * self.C[i][j] + self.c1 * self.pc[i] * self.pc[j] + self.cmu * cmuij

        #        for i in range(self.nn):
        #            for j in range(self.nn):
        #                f.write(str(self.C[i][j]) + ' ')
        #            f.write('\n')

        # update step-size
        self.sigma *= np.exp(min(0.6, (self.cs / self.damps) * (sum(x ** 2 for x in self.ps) / self.nn - 1) / 2))

    def run(self):
        for i in range(self.maxgen):
            self.step()
            print(i, self.opti_f)

    def output(self):
        return self.opti_f, self.opti_x
