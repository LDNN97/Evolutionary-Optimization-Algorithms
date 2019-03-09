import numpy as np


class Sphere50D:
    D = 50
    lb, ub = -1, 1

    @staticmethod
    def evaluate(x):
        x = Sphere50D.lb + x * (Sphere50D.ub - Sphere50D.lb)
        re = sum(np.power(x[i], 2) for i in range(Sphere50D.D))
        return re


class Rosenbrock50D:
    D = 50
    lb, ub = -50, 50

    @staticmethod
    def evaluate(x):
        x = Rosenbrock50D.lb + x * (Rosenbrock50D.ub - Rosenbrock50D.lb)
        re = 0
        for i in range(Rosenbrock50D.D - 1):
            re += 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2
        return re


class Ackley50D:
    D = 50
    lb, ub = -50, 50

    @staticmethod
    def evaluate(x):
        x = Ackley50D.lb + x * (Ackley50D.ub - Ackley50D.lb)

        # shift operation
        # x = x - 42.0969

        part1 = 0
        for i in range(Ackley50D.D):
            part1 += x[i] ** 2
        part2 = 0
        for i in range(Ackley50D.D):
            part2 += np.cos(2 * np.pi * x[i])
        re = -20 * np.exp(-0.2 * np.sqrt(part1 / Ackley50D.D)) \
            - np.exp(part2 / Ackley50D.D) + 20 + np.e
        return re


class Rastrgin50D:
    D = 50
    lb, ub = -50, 50

    @staticmethod
    def evaluate(x):
        x = Rastrgin50D.lb + x * (Rastrgin50D.ub - Rastrgin50D.lb)
        re = 0
        for i in range(Rastrgin50D.D):
            re += x[i] ** 2 - 10 * np.cos(2 * np.pi * x[i]) + 10
        return re


class Griewank50D:
    D = 50
    lb, ub = -100, 100

    @staticmethod
    def evaluate(x):
        x = Griewank50D.lb + x * (Griewank50D.ub - Griewank50D.lb)
        part1, part2 = 0, 1
        for i in range(Griewank50D.D):
            part1 += x[i] ** 2
            part2 *= np.cos(x[i] / np.sqrt(i + 1))
        re = 1 + part1 / 4000 - part2
        return re


class Weierstrass50D:
    D = 50
    lb, ub = -0.5, 0.5

    @staticmethod
    def evaluate(x):
        x = Weierstrass50D.lb + x * (Weierstrass50D.ub - Weierstrass50D.lb)
        part1 = 0
        for i in range(Weierstrass50D.D):
            for j in range(21):
                part1 += np.power(0.5, j) * np.cos(2 * np.pi * np.power(3, j) * (x[i] + 0.5))
        part2 = 0
        for i in range(21):
            part2 += np.power(0.5, i) * np.cos(2 * np.pi * np.power(3, i) * 0.5)
        re = part1 - Weierstrass50D.D * part2
        return re


class Schwefel50D:
    D = 50
    lb, ub = -500, 500

    @staticmethod
    def evaluate(x):
        x = Schwefel50D.lb + x * (Schwefel50D.ub - Schwefel50D.lb)
        part1 = 0
        for i in range(Schwefel50D.D):
            part1 += x[i] * np.sin(np.sqrt(np.abs(x[i])))
        re = 418.9829 * Schwefel50D.D - part1
        return re
