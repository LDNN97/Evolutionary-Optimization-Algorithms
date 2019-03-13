import numpy as np
from .optimizer import *


class CMAES(Optimizer):
    def __init__(self, func, maxgen, maxeval, pop_size, s_size):
        super().__init__(func, maxgen)

        self.nn = self.f.D
        self.xx = np.random.random(self.nn)
        self.xmean = self.xx[:]
        self.sigma = s_size

        self.lam = pop_size
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
        self.M = np.eye(self.nn)

        self.evals = 0
        self.maxeval = maxeval
        self.gen_fit = []
        self.tc = False

    def step(self):
        # Sample
        self.D, self.B = np.linalg.eigh(self.C)
        self.D = self.D ** 0.5
        self.M = self.B * self.D
        newpop, z, d, fitvals = [], [], [], []
        for i in range(self.lam):
            zz = np.random.normal(0, 1, self.nn)
            dd = np.dot(self.M, zz)
            nn = self.xmean + self.sigma * dd

            # check boundary
            for j in range(len(nn)):
                if nn[j] > 1:
                    nn[j] = self.xmean[j] + np.random.random() * (1 - self.xmean[j])
                if nn[j] < 0:
                    nn[j] = np.random.random() * (self.xmean[j])
            dd = (nn - self.xmean) / self.sigma
            invm = np.linalg.inv(self.M)
            zz = np.dot(invm, dd)

            z.append(zz), d.append(dd), newpop.append(nn)

        # sort and update mean
        fitvals = []
        for xx in newpop:
            self.evals += 1

            if self.evals >= self.maxeval:
                self.tc = True

            fit = self.f.evaluate(xx)
            fitvals.append(fit)
        argx = np.argsort(fitvals)
        if fitvals[argx[0]] < self.opti_f:
            self.opti_x = newpop[argx[0]]
            self.opti_f = fitvals[argx[0]]
        self.xmean = sum(self.weights[i] * newpop[argx[i]] for i in range(self.mu))  # Important

        deltax = self.sigma * d[argx[0]]
        dist = np.linalg.norm(deltax)
        if dist < 1e-7:
            self.tc = True

        # update evolution path
        zz = sum(self.weights[i] * z[argx[i]] for i in range(self.mu))
        c = (self.cs * (2 - self.cs) * self.mueff) ** 0.5
        self.ps -= self.cs * self.ps
        self.ps += c * zz
        dd = sum(self.weights[i] * d[argx[i]] for i in range(self.mu))
        c = (self.cc * (2 - self.cc) * self.mueff) ** 0.5
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
        self.sigma *= np.exp(min(0.6, (self.cs / self.damps) * (sum(x ** 2 for x in self.ps) / self.nn - 1) / 2))

    def run(self):
        for i in range(self.maxgen):
            self.step()

            print(i, "%.6f" % self.opti_f)

            self.gen_fit.append(self.opti_f)
            if i > 9:
                if self.gen_fit[i - 20] - self.gen_fit[i] < 0.000001:
                    self.tc = True

            if self.tc:
                break

    def output(self):
        return self.opti_f, self.opti_x


class CMAESB(Optimizer):
    def __init__(self, func, maxgen, maxeval):
        super().__init__(func, maxgen)
        self.max_eval = maxeval
        self.pop_size = 4 + int(3 * np.log(self.f.D))
        self.s_size = 1 / np.sqrt(self.f.D)

        self.opti_f = 1e10
        self.opti_x = []

    def update(self, f, x):
        if f < self.opti_f:
            self.opti_f = f
            self.opti_x = x

    def output(self):
        return self.opti_f, self.opti_x

    def run(self):
        print("Normal", self.pop_size)
        cma_normal = CMAES(self.f, self.maxgen, self.max_eval, self.pop_size, self.s_size)
        cma_normal.run()
        self.update(cma_normal.opti_f, cma_normal.opti_x)
        n, n_s = 0, 0
        b_1 = cma_normal.evals
        b_l, b_s = 0, 0
        while True:
            n = n + 1
            pop_size = np.power(2, n - n_s) * self.pop_size
            if n > 2 and b_s < b_l:
                s_size = self.s_size / np.power(100, np.random.random())
                pops_size = int(self.pop_size * np.power((pop_size / (2 * self.pop_size)), np.random.random() ** 2))
                gen = int(b_l / (2 * pops_size))
                print("Local", pops_size)
                cma_local = CMAES(self.f, gen, self.max_eval, pops_size, s_size)
                cma_local.run()
                self.update(cma_local.opti_f, cma_local.opti_x)
                b_s += cma_local.evals
                n_s += 1
            else:
                print("Global", pop_size)
                cma_global = CMAES(self.f, self.maxgen, self.max_eval, pop_size, self.s_size)
                cma_global.run()
                self.update(cma_global.opti_f, cma_global.opti_x)
                b_l += cma_global.evals
            if b_1 + b_s + b_l >= self.max_eval:
                break
