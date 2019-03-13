# Evolutionary Optimization Algorithms

[![Build Status](https://travis-ci.com/LDNN97/Evolutionary-Optimization-Algorithms.svg?branch=master)](https://travis-ci.com/LDNN97/Evolutionary-Optimization-Algorithms) [![codecov](https://codecov.io/gh/LDNN97/Evolutionary-Optimization-Algorithms/branch/master/graph/badge.svg)](https://codecov.io/gh/LDNN97/Evolutionary-Optimization-Algorithms) 

Papers:

DE: [Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces](https://link.springer.com/article/10.1023/A:1008202821328)

CMA-ES: [The CMA Evolution Strategy: A Tutorial](https://arxiv.org/pdf/1604.00772)

Restart CMA-ES: [A Restart CMA Evolution Strategy With Increasing Population Size](https://ieeexplore.ieee.org/abstract/document/1554902/)

MA-ES: [Simplify your covariance matrix adaptation evolution strategy](https://ieeexplore.ieee.org/abstract/document/7875115/)

LM-CMA: [LM-CMA: An alternative to L-BFGS for large-scale black box optimization](https://www.mitpressjournals.org/doi/abs/10.1162/EVCO_a_00168)

LM-MA: [Large Scale Black-box Optimization by Limited-Memory Matrix Adaptation](https://ieeexplore.ieee.org/abstract/document/8410043/)

ES for RL: [Evolution Strategies as A Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)

## Usage

1. clone the repository

2. `import eoa`

3. select a problem already existed or define your only problem

4. select an algorithm to find the optimum.

You can look into `eoa.py` to find more information about the usage.

## Example

``` python
TaskProb = Sphere(50, -50, 50)
Task = DE(TaskProb, 1000)
Task.run()
```

### Class for optimizer

``` python
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
```

### Class for problem

``` python
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
```
