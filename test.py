import time
import numpy as np

a = []

s = time.clock()

for i in range(1000000):
    a.append(i)
print(type(a))

b = a[:]
e = time.clock()

print(e - s)

a = np.zeros(1000000)

s = time.clock()

for i in range(1000000):
    a[i] = i

b = np.copy(a)

e = time.clock()


print(e - s)




