import problems

from de import DE
from cmaes import CMAES
from cmaes_origin import CMAESO
from cmaes_bipop import CMAESB
from cmaes_maes import CMAESM
from cmaes_large import CMAESL

if __name__ == "__main__":
    TaskProb = problems.Sphere(50, -50, 50)
    Task = DE(TaskProb, 1000)
    Task.run()
