from prob.problems import *

from opti.de import DE
from opti.cmaes import CMAES
from opti.cmaes_maes import CMAESM
from opti.cmaes_large import CMAESL


class TestClass(object):
    def test_1(self):
        task_p = Sphere(50, -50, 50)
        task = DE(task_p, 1000)
        task.run()
        assert task.opti_f < 1

    def test_2(self):
        task_p = Sphere(50, -50, 50)
        task = CMAES(task_p, 1000)
        task.run()
        assert task.opti_f < 1

    def test_3(self):
        task_p = Sphere(50, -50, 50)
        task = CMAESM(task_p, 1000)
        task.run()
        assert task.opti_f < 1

    def test_4(self):
        task_p = Sphere(50, -50, 50)
        task = CMAESL(task_p, 1000)
        task.run()
        assert task.opti_f < 1
